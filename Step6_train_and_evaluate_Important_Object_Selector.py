######### Step6_train_and_evaluate_Important_Object_Selector.py
### Note: Risk-objects referenced in the code is synonymous with Important Objects
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, copy, random
from collections import Counter
from mmaction.apis import init_recognizer, inference_recognizer
from mmcv.ops import roi_align
import mmcv
from collections import Counter
from mmcv.fileio import FileClient
import os.path as osp
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
torch.random.manual_seed(1)
mmcv.use_backend('cv2')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

root_dir = 'IDD-D_Yolo-v4-Model/'
namesfile = root_dir + 'idd.names'
mot_classes = load_class_names(namesfile)


### Load train-val dataset of IOs
dstype = 'train'
trainset_labels_dist = pd.read_csv('riskobj_labels_MOTwise_dist_'+dstype+'.csv')
labels_order = np.array(trainset_labels_dist['labels'])
num_classes = len(labels_order)
print("Total classes:",num_classes)

trainset = np.load('pertrack_'+dstype+'_data_motwise_dict.npy', allow_pickle=True).tolist()
trainset_labels = trainset['riskobj_class']
trainset_labels = np.array(trainset_labels)
print("Training set:", len(trainset_labels))

dstype = 'val'
valset = np.load('pertrack_'+dstype+'_data_motwise_dict.npy', allow_pickle=True).tolist()
valset_labels = valset['riskobj_class']
valset_labels = np.array(valset_labels)
print("Validation set:", len(valset_labels))


### Class-balanced mini-batch data loader
def prepLabelsDS(all_labels, num_samples=None, batch_size = 15):   ### batch size should be a multiple of N -> no. of classes
    S = dict()
    c_max = 0
    for idx in range(len(all_labels)):
        label = all_labels[idx]
        if label not in S:
            S[label] = list()
        S[label].append(idx)
        if(len(S[label]) > c_max):
            c_max = len(S[label])
    N = len(S)
    if(num_samples is not None):
        c_max = num_samples//N
    Sbkp = copy.deepcopy(S)
    return Sbkp, c_max    

def getClassBalancedBatches(Sbkp, c_max, batch_size = 15):   
    ### do this for every epoch
    N = len(Sbkp)
    num_batches = N*(c_max//batch_size)
    S = copy.deepcopy(Sbkp)
    ### oversample with shuffling
    for k in S:
        if(len(S[k]) < c_max):
            temp = []
            for ai in range(0, c_max//len(S[k])):
                random.shuffle(S[k])
                temp = temp + S[k]
            temp = temp + random.sample(S[k], c_max % len(S[k]))
            S[k] = temp
        else:
            S[k] = random.sample(S[k], c_max)
    ### preparing batches
    B = []
    for i in range(num_batches):
        Bi = []
        for k in S:
            start_idx = int(i*batch_size//N)
            end_idx = int((i+1)*batch_size//N)
            Bi += S[k][start_idx:end_idx]
        random.shuffle(Bi)
        B.append(Bi)
    random.shuffle(B)
    return B

batch_size = 32
## Sbkp contains class-specific dataset-indices list; c_max_samples is the no. of samples to retrieve from each class (by over/under-sampling)
# Sbkp, c_max_samples = prepLabelsDS(trainset_labels, num_samples=int(len(trainset_labels)/10), batch_size = batch_size)
Sbkp, c_max_samples = prepLabelsDS(trainset_labels, num_samples=int(len(trainset_labels)/11), batch_size = batch_size)
print("max samples per class after class-balancing:", c_max_samples)
class_counts = {}
for skey in Sbkp:
    class_counts[skey] = len(Sbkp[skey])
print("Class counts:", class_counts)


##### Encoding mot_obj_class category values between 0-1
trainset['merged_motobj_classes'] = (np.array(trainset['motobj_label']) + 1) / len(mot_classes)
valset['merged_motobj_classes'] = (np.array(valset['motobj_label']) + 1) / len(mot_classes)
print(len(trainset['merged_motobj_classes']), np.unique(trainset['merged_motobj_classes']))
print(len(valset['merged_motobj_classes']), np.unique(valset['merged_motobj_classes']))

trainset_motobj_labels = np.array(trainset['motobj_label'])
valset_motobj_labels = np.array(valset['motobj_label'])


image_width = 1920
image_height = 1080

### Control the amount of FPS jitter (should always be > 1)
max_fps_jitter_factor = 3.0
def get_fps_jitter_fortrack(track_duration, scene_duration, max_fps_jitter_factor=max_fps_jitter_factor):
    return np.interp(track_duration, np.array([1, scene_duration]), np.array([1/max_fps_jitter_factor, max_fps_jitter_factor]))

def trackboxes_fpsjitter(temp_bbox, fps_jitter_factor = 1.0):
    original_locations = np.arange(0, temp_bbox.shape[0], 1)
    interp_locations = np.arange(0, temp_bbox.shape[0], fps_jitter_factor)
    interp_locations = interp_locations[interp_locations<=original_locations[-1]]

    interpolated_trackboxes = np.zeros((len(interp_locations), 4))
    interpolated_trackboxes[:, 0] = np.interp(interp_locations, original_locations, temp_bbox[:, 0])
    interpolated_trackboxes[:, 1] = np.interp(interp_locations, original_locations, temp_bbox[:, 1])
    interpolated_trackboxes[:, 2] = np.interp(interp_locations, original_locations, temp_bbox[:, 2])
    interpolated_trackboxes[:, 3] = np.interp(interp_locations, original_locations, temp_bbox[:, 3])
    return(interpolated_trackboxes)

### Data loader with Augmentations
def batch_seq_loader(batch_idxs, dataset_npy, dstype='train'):
    flip_aug = False
    input_seq = []
    input_seq_lengths = []
    # input_seq_lengths_ = []
    # input_seq_labelbinh0s = torch.Tensor()
    input_seq_labelbinh0s = []
    
    for ebi in batch_idxs:
        ### apply flip aug
        flip_aug = True if(random.random()>0.5 and dstype=='train') else False
        
        stt, endt = dataset_npy['video_dir'][ebi].split('_')[-1].split('to')
        track_scene_duration = int(endt) - int(stt) + 1

        bboxis = dataset_npy['track_bboxes'][ebi]
        ## can be optimized by pre-computing values
        obj_class_idx_normalized = dataset_npy['merged_motobj_classes'][ebi]
        ###################### interpolate if missing bboxes
        input_feats = []
        for bxi, bboxi in enumerate(bboxis):
            if(bxi>0):
                ### diff in frame nos should be 1.
                fnodiff = bboxi[-1] - bboxis[bxi-1][-1]
                if((fnodiff) > 1):
                    for fi in range(int(bboxis[bxi-1][-1]+1), int(bboxi[-1])):
                        w1, w2 = ((fi-bboxis[bxi-1][-1])/fnodiff , (bboxi[-1]-fi)/fnodiff)
                        xcenter1, ycenter1, wd1, ht1 = [(bboxi[0]+bboxi[2])/2, (bboxi[1]+bboxi[3])/2, (bboxi[2]-bboxi[0]), (bboxi[3]-bboxi[1])]
                        xcenter2, ycenter2, wd2, ht2 = [(bboxis[bxi-1][0]+bboxis[bxi-1][2])/2, (bboxis[bxi-1][1]+bboxis[bxi-1][3])/2, (bboxis[bxi-1][2]-bboxis[bxi-1][0]), (bboxis[bxi-1][3]-bboxis[bxi-1][1])]
                        ### avg. interpolate and normalize
                        # input_feats.append([(w1*xcenter1+w2*xcenter2)/image_width, (w1*ycenter1+w2*ycenter2)/image_height, (w1*wd1+w2*wd2)/image_width, (w1*ht1+w2*ht2)/image_height])
                        input_feats.append([(w1*xcenter1+w2*xcenter2)/image_width, (w1*ycenter1+w2*ycenter2)/image_height, (w1*wd1+w2*wd2)/image_width, (w1*ht1+w2*ht2)/image_height, obj_class_idx_normalized])
                        # input_feats.append([(w1*xcenter1+w2*xcenter2)/image_width, (w1*ycenter1+w2*ycenter2)/image_height, (w1*wd1+w2*wd2)/image_width, (w1*ht1+w2*ht2)/image_height] + obj_class_idx_binh0)
                        if(flip_aug):
                            input_feats[-1][0] = 1 - input_feats[-1][0] - input_feats[-1][2]
            xcenter, ycenter, wd, ht = [(bboxi[0]+bboxi[2])/2, (bboxi[1]+bboxi[3])/2, (bboxi[2]-bboxi[0]), (bboxi[3]-bboxi[1])]
            ### normalize
            # input_feats.append([xcenter/image_width, ycenter/image_height, wd/image_width, ht/image_height])
            input_feats.append([xcenter/image_width, ycenter/image_height, wd/image_width, ht/image_height, obj_class_idx_normalized])
            # input_feats.append([xcenter/image_width, ycenter/image_height, wd/image_width, ht/image_height] + obj_class_idx_binh0)
            if(flip_aug):
                input_feats[-1][0] = 1 - input_feats[-1][0] - input_feats[-1][2]

        input_feats = np.array(input_feats, dtype=np.float32)
        ### Random FPS Jitter augmentation for track boxes
        # if(False):
        if(random.random()>0.5 and dstype=='train'):
            ### getting the fps jitter value for the current track length with respect to the total scene duration.
            track_fpsjitter = get_fps_jitter_fortrack(input_feats.shape[0], track_scene_duration)
            ### getting the interpolated bbox coords and width,hts for the fps jittered track
            modified_trackboxes = trackboxes_fpsjitter(input_feats[:, :4], fps_jitter_factor = track_fpsjitter)
            ### concatenated obj class as last column
            input_feats = np.concatenate([modified_trackboxes, np.array([[obj_class_idx_normalized]*modified_trackboxes.shape[0]]).transpose()], axis=-1)
        
        input_seq.append(torch.from_numpy(input_feats).type(torch.FloatTensor))
        # input_seq.append(torch.tensor(input_feats, dtype=torch.float32))
        # input_seq.append(torch.tensor(bboxis))
        input_seq_lengths.append(len(input_seq[-1]))
        # input_seq_lengths_.append(int(bboxis[-1,-1]-bboxis[0,-1]+1))
        # input_seq_labelbinh0s = torch.cat([input_seq_labelbinh0s, obj_class_idx_binh0],dim=1)
        # input_seq_labelbinh0s.append( obj_class_idx_binh0 )
    
    # input_seq_labelbinh0s = torch.tensor(input_seq_labelbinh0s, dtype=torch.float32)
    # input_seq_labelbinh0s = torch.nn.functional.one_hot(torch.tensor(dataset_npy['motobj_label']), num_classes=len(mot_classes)).type(torch.FloatTensor)
    input_seq_labelbinh0s = torch.from_numpy(np.array(dataset_npy['motobj_label'])[batch_idxs])
    
    # return(torch.nn.utils.rnn.pad_sequence(input_seq, batch_first=True), input_seq_lengths, None)
    return(torch.nn.utils.rnn.pad_sequence(input_seq, batch_first=True), input_seq_lengths, input_seq_labelbinh0s)


### Important Object Identification model definition (bidirectional GRU with object class conditioning)
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # ### len(mot_classes) = 15
        self.fc_labels = nn.Embedding(15, input_size)
        # self.fc_labels = nn.Embedding(15, hidden_size)
        # self.fc_labels = nn.Embedding(15, 1)
        
    def forward(self, x, x_onehotlabels, x_seq_lengths, h0=None):
        
        # v_vec = self.fc_labels(x_onehotlabels)
        # # x = torch.cat([v_vec.unsqueeze(1), x], dim=1)
        # x = torch.cat([x, v_vec.unsqueeze(1)], dim=1)
        
        if(h0 is None):
            ### M1:
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
            h0 = h0.to(x.device)
            ### M2:
            # h0 = torch.cat([v_vec.unsqueeze(0), v_vec.unsqueeze(0)], dim=0)
            # h0 = h0.to(x.device)
            ### M3:
            # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # h0 = torch.cat([v_vec.unsqueeze(0), h0], dim=0)
        
        ### if no hidden state is passed then by default they are intialized  to zeros.
        # out, _ = self.gru(x)
        out, hidden_st = self.gru(x, h0)
        
        # out, out_seq_lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        # ### no help:
        # out = self.relu(out)
        
        out = self.fc(out[:, -1, :])

        return out

hidden_size = 5
num_layers = 1
input_feature_size = 5
nlabels = 2
#### nlabels and input_feature_size defined above
model = BRNN(input_feature_size, hidden_size, num_layers, nlabels).to(device)

# wandb.config = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size, "hidden_size":hidden_size, "num_layers":num_layers, 'no_of_features':no_of_features,'max_no_of_objects':max_no_of_objects}#, "input_feature_size":input_feature_size}
# wandb.watch(model, log_freq=100)

learning_rate = 0.001 #0.01
# Loss and optimizer
# criteria = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 1.2]).to(device))
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.25)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.35)


### training pipeline
log_dir = 'work_dirs/logs/'
os.makedirs(log_dir, exist_ok=True)
model_dir = 'work_dirs/saved_models/'
os.makedirs(model_dir, exist_ok=True)
###
model_name = 'iddx_riskobj_id_rnn_on_idddtracks_lr0.001_wflipaug_wobjclass_e100'
print("Model name:",model_name)

exp_df = {'epoch':[], 'NonRisk_recall': [], 'Risk_recall': [], 'NonRisk_fscore': [], 'Risk_fscore': [], 'AvgAcc':[], 'top1_acc':[], 'epoch_acc':[], 'loss':[]}

### for best val accuracy
best_loss = 0
num_epochs = 100
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            epoch_batches = getClassBalancedBatches(Sbkp, c_max_samples, batch_size = batch_size)
        else:
            model.eval()
            epoch_batches = [list(range(batch_idx, min(batch_idx+batch_size, len(valset_labels)))) for batch_idx in range(0, len(valset_labels), batch_size)]
        num_batches_ = len(epoch_batches)
        
        pred_labels = []
        running_loss = 0.0
        running_acc = 0
        n_examples = 0
        for iteration, batch_idxs in enumerate(epoch_batches):
            batched_data = None
            if phase == 'train':
                batch_seq, batch_seq_lengths, batch_seq_labelbinh0s = batch_seq_loader(batch_idxs, trainset, dstype=phase)
                batch_labels = torch.tensor(trainset_labels[batch_idxs]).to(device)
                batch_seq = batch_seq.to(device)
                batch_seq_labelbinh0s = batch_seq_labelbinh0s.to(device)
            else:
                batch_seq, batch_seq_lengths, batch_seq_labelbinh0s = batch_seq_loader(batch_idxs, valset, dstype=phase)
                batch_labels = torch.tensor(valset_labels[batch_idxs]).to(device)
                batch_seq = batch_seq.to(device)
                batch_seq_labelbinh0s = batch_seq_labelbinh0s.to(device)
            
            ### forward pass
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # output = model(batch_seq, batch_seq_lengths)
                # output = model(batch_seq, batch_seq_lengths, h0 = batch_seq_labelbinh0s)
                # output = model(batch_seq, batch_onehotmotlabels, batch_seq_lengths, h0 = batch_seq_labelbinh0s)
                output = model(batch_seq, batch_seq_labelbinh0s, batch_seq_lengths, h0 = None)
                _, pred = torch.max(output, 1)
                loss = criteria(output, batch_labels)
                # loss = torchvision.ops.focal_loss.sigmoid_focal_loss(output, torch.nn.functional.one_hot(batch_labels, num_classes=2).type(torch.FloatTensor).to(device), alpha=-1, gamma=0.75, reduction='mean')
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
                    optimizer.step()
                
            running_loss += loss.item()
            running_acc += (batch_labels.cpu() == pred.cpu()).sum()
            n_examples += batch_labels.shape[0]
            if phase == 'val':
                if pred_labels is None:
                    pred_labels = pred.cpu().numpy()
                else:
                    pred_labels = np.concatenate((pred_labels, pred.cpu().numpy()), -1)

            # Print the average loss in a mini-batch.
            if iteration % 50 == 0:
                print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(phase.upper(), epoch+1, num_epochs, iteration, num_batches_, loss.item()))
            
        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / num_batches_
        epoch_acc = running_acc.double() / n_examples

        print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f} \n'.format(phase.upper(), epoch+1, num_epochs, epoch_loss, epoch_acc))
        
        if phase == 'val':
            performance_out = classification_report(valset_labels, list(pred_labels), target_names= ['NotRisk', 'Risk'], zero_division=1)

            nonrisk_acc, risk_acc, nonrisk_fscore, risk_fscore, avg_acc, wavg_acc = \
            float(performance_out.split('\n')[2].strip().split('     ')[2].strip()), \
            float(performance_out.split('\n')[3].strip().split('     ')[2].strip()), \
            float(performance_out.split('\n')[2].strip().split('     ')[3].strip()), \
            float(performance_out.split('\n')[3].strip().split('     ')[3].strip()), \
            float(performance_out.split('\n')[6].strip().split('     ')[2].strip()), \
            float(performance_out.split('\n')[7].strip().split('     ')[2].strip())
            print(nonrisk_acc, risk_acc, nonrisk_fscore, risk_fscore, avg_acc, wavg_acc)

            exp_df['epoch'].append(epoch+1)
            exp_df['loss'].append(epoch_loss)
            exp_df['epoch_acc'].append(epoch_acc.cpu().numpy())
            exp_df['top1_acc'].append(wavg_acc)
            exp_df['AvgAcc'].append(avg_acc)
            exp_df['NonRisk_recall'].append(nonrisk_acc)
            exp_df['Risk_recall'].append(risk_acc)
            exp_df['NonRisk_fscore'].append(nonrisk_fscore)
            exp_df['Risk_fscore'].append(risk_fscore)
        else:
            # scheduler.step()
            print("Learning rate:", scheduler.get_last_lr())

        # Log the loss and accuracy in an epoch.
        with open(os.path.join(log_dir, '{}-{}-log-epoch-{:02}.txt').format(model_name, phase, epoch+1), 'w') as f:
            f.write(str(epoch+1) + '\t'
                    + str(epoch_loss) + '\t'
                    + str(epoch_acc.item()) + '\t'
                    + str(0))

        if phase == 'val':
            # acc_fscore = (2*avg_acc * wavg_acc)/(avg_acc + wavg_acc)
            acc_fscore = (2*risk_acc * risk_fscore)/(risk_acc + risk_fscore)
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(model_dir, '{}-ep{}.ckpt'.format(model_name, str(epoch+1))))
            # if epoch_loss < best_loss:
            # if epoch_acc > best_loss:
            if acc_fscore > best_loss:
                # print("At epoch:",epoch+1,"best loss from:\t",best_loss, "\tto\t",epoch_loss)
                # print("At epoch:",epoch+1,"best acc. from:\t",best_loss, "\tto\t",epoch_acc)
                print("At epoch:",epoch+1,"best acc. from:\t",best_loss, "\tto\t",acc_fscore, " \tAverage and Wacg Acc.:", avg_acc , wavg_acc, risk_acc , risk_fscore)
                # best_loss = epoch_loss
                # best_loss = epoch_acc
                best_loss = acc_fscore
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(model_dir, '{}-best_model.ckpt'.format(model_name)))
    

exp_df = pd.DataFrame(exp_df)
exp_df.to_csv(model_name.split('best_model')[0]+'all_epoch_models_accuracies.csv',index=False)
print(exp_df)

###################################### End of Training ######################################


### Evaluating the best model
model_name_eval = os.path.join(model_dir, '{}-best_model.ckpt'.format(model_name))
# model_name_eval = os.path.join(model_dir, '{}-best_model.ckpt'.format(model_name.split('-best_model')[0].split('work_dirs/saved_models/')[1]))
print("Validating performance for: ", model_name_eval)
model.load_state_dict(torch.load(model_name_eval, map_location=device)['state_dict'])
model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    for phase in ['val']:
        if phase == 'train':
            model.train()
            epoch_batches = getClassBalancedBatches(Sbkp, c_max_samples, batch_size = batch_size)
        else:
            model.eval()
            epoch_batches = [list(range(batch_idx, min(batch_idx+batch_size, len(valset_labels)))) for batch_idx in range(0, len(valset_labels), batch_size)]
        num_batches_ = len(epoch_batches)
        
        pred_labels = []
        running_loss = 0.0
        running_acc = 0
        n_examples = 0
        for iteration, batch_idxs in enumerate(epoch_batches):
            batched_data = None
            if phase == 'train':
                batch_seq, batch_seq_lengths, batch_seq_labelbinh0s = batch_seq_loader(batch_idxs, trainset, dstype=phase)
                batch_labels = torch.tensor(trainset_labels[batch_idxs]).to(device)
                batch_seq = batch_seq.to(device)
                batch_seq_labelbinh0s = batch_seq_labelbinh0s.to(device)
            else:
                batch_seq, batch_seq_lengths, batch_seq_labelbinh0s = batch_seq_loader(batch_idxs, valset, dstype=phase)
                batch_labels = torch.tensor(valset_labels[batch_idxs]).to(device)
                batch_seq = batch_seq.to(device)
                batch_seq_labelbinh0s = batch_seq_labelbinh0s.to(device)
                
            ### forward pass
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # output = model(batch_seq, batch_seq_lengths)
                # output = model(batch_seq, batch_seq_lengths, h0 = batch_seq_labelbinh0s)
                # output = model(batch_seq, batch_onehotmotlabels, batch_seq_lengths, h0 = batch_seq_labelbinh0s)
                output = model(batch_seq, batch_seq_labelbinh0s, batch_seq_lengths, h0 = None)
                _, pred = torch.max(output, 1)
                loss = criteria(output, batch_labels)
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
                    optimizer.step()

            running_loss += loss.item()
            running_acc += (batch_labels.cpu() == pred.cpu()).sum()
            n_examples += batch_labels.shape[0]
            if phase == 'val':
                if pred_labels is None:
                    pred_labels = pred.cpu().numpy()
                else:
                    pred_labels = np.concatenate((pred_labels, pred.cpu().numpy()), -1)

            # Print the average loss in a mini-batch.
            if iteration % 50 == 0:
                print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(phase.upper(), epoch+1, num_epochs, iteration, num_batches_, loss.item()))
            
        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / num_batches_
        epoch_acc = running_acc.double() / n_examples

        print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f} \n'.format(phase.upper(), epoch+1, num_epochs, epoch_loss, epoch_acc))
        
        if phase == 'val':
            performance_out = classification_report(valset_labels, list(pred_labels), target_names= ['NotRisk', 'Risk'], zero_division=1,digits=3)


print("################# Best Model Evaluation Results for Important Object Track Identification #################")
print(performance_out)
# disp = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(valset_labels, list(pred_labels)))
# disp.plot()