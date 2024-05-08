######### Step13_train_IOeXplanationsGeneratorModel.py
### Note: Risk-objects referenced in the code is synonymous with Important Objects

# ### important scripts to consider before training X model
# tsn_config-activitynet-rgb_iddx_latest_wCorrectCropAug_testforXmodeltrain.py
# mmaction2/mmaction/apis/inference.py
# mmaction2/mmaction/datasets/pipelines/loading.py
# mmaction2/mmaction/datasets/pipelines/augmentations.py
### in loading.py: ### it does replicate padding otherwise (which maybe good for shorter video sequences, but the current pre-processing pipeline for X-model this replicate padding is not handled ---> rectify)
### total_frames = results['total_frames']
### total_frames = self.clip_len*(results['total_frames']//self.clip_len)   #results['total_frames']

import torch
import torch.nn as nn
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
mmcv.use_backend('cv2')

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

### Load dataset
dstype = 'train'
trainset_labels_dist = pd.read_csv('riskobj_explanation_labels_dist_'+dstype+'_wextras.csv')
labels_order = np.array(trainset_labels_dist['labels'])
num_classes = len(labels_order)
print("Total explanation classes:",num_classes)

trainset = np.load('pertrack_'+dstype+'_data_atpmerged_dict_wextras.npy', allow_pickle=True).tolist()
trainset_labels = []
for li in trainset['explanation_class']:
    trainset_labels.append(np.where(li==labels_order)[0][0])
trainset_labels = np.array(trainset_labels)
print("Training set:", len(trainset_labels))

dstype = 'val'
valset = np.load('pertrack_'+dstype+'_data_atpmerged_dict_wextras.npy', allow_pickle=True).tolist()
valset_labels = []
for li in valset['explanation_class']:
    valset_labels.append(np.where(li==labels_order)[0][0])
valset_labels = np.array(valset_labels)
print("Validation set:", len(valset_labels))

device = 'cuda:0'
# device = 'cpu'
device = torch.device(device)

### Load the best model obtained after running 'Step9_evaluate_tsn_egobehavrecog.py'
checkpoint_file = 'models/tsn_flow_IDDX_EgoBehaviorRecognizer.pth'

### The 'test' pipeline has been modified (a. randomized sampling of frames, b. no periphery cropping, c. added flip augmentation)
config_file = 'tsn_config-activitynet-rgb_iddx_latest_wCorrectCropAug_testforXmodeltrain.py'

### build the model from a config file and a checkpoint file
ar_model = init_recognizer(config_file, checkpoint_file, device=device)

def get_ar_feats(video_dir, track_bboxes, dstype, temporal_window_range = (0.75, 1.0), modality = 'Flow', filename_tmpl = 'flow_{}_{:05d}.jpg', offset = 0, clip_length = 5):
    video_src_dir = 'iddx_clips_rawframe_front/'+dstype+'_flow/Scenes/'
    video = video_src_dir+video_dir

    ### reducing the size by 1 since optical flow has 1 less frame
    track_bboxes = track_bboxes[:-1,:]

    ### Randomized Temporal-window-cropping
    tlen = len(track_bboxes)
    twindow_ratio = np.random.uniform(temporal_window_range[0], temporal_window_range[1])
    toffset = np.random.randint(0, np.max((tlen*(1-twindow_ratio), 1)) )
    t_start_idx, t_end_idx = (toffset, toffset + round(tlen*twindow_ratio))    ## indices w.r.t. track-bboxes

    tcropped_track_bboxes = track_bboxes[t_start_idx:t_end_idx, :]
    frame_start_idx, frame_end_idx = (tcropped_track_bboxes[0, 4], tcropped_track_bboxes[-1, 4])    ## absolute video-frame-nos (for cropped temporal-window)

    timeframe_range = video.split('_')[-1].split('to')
    # frame_inds = range(0, int(timeframe_range[1]) - int(timeframe_range[0]) + 1)
    
    video_start_idx = int(frame_start_idx) - int(timeframe_range[0])
    video_end_idx = int(frame_end_idx) - int(timeframe_range[0])
    frame_inds = range(video_start_idx, video_end_idx + 1)    ## relative video-frame-nos w.r.t. the actual starting-point of absolute video-frame-no 

    file_client = FileClient('disk')
    imgs = list()
    for i, frame_idx in enumerate(frame_inds):
        frame_idx += offset
        if modality == 'Flow':
            x_filepath = osp.join(video, filename_tmpl.format('x', frame_idx))
            y_filepath = osp.join(video, filename_tmpl.format('y', frame_idx))
            x_img_bytes = file_client.get(x_filepath)
            x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
            y_img_bytes = file_client.get(y_filepath)
            y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
            # imgs.extend([x_frame, y_frame])
            imgs.extend([np.stack((x_frame, y_frame), axis = -1)])
        else:
            raise NotImplementedError

    ### a 4D array: T(time-frames) x H x W x C(=2 for optical flow)
    imgs = np.stack(imgs)
    
    tracks_rois = None
    n_trys = 0
    while(tracks_rois is None):
        n_trys += 1
        results, ret_feats, img_metas = inference_recognizer(ar_model, imgs, outputs=['backbone.layer4', 'cls_head', 'cls_head.consensus', 'cls_head.avg_pool'], dstype=dstype)
        img_metas = img_metas.data[0][0]

        img_flip = img_metas['flip']
        final_img_width = img_metas['img_shape'][1]
        bbox_xyxy_scaling = np.concatenate((img_metas['scale_factor'], img_metas['scale_factor']))

        ### clip_centers corresponds to clip's center-frame relative (w.r.t. video_start_idx) indices, required only clip-len from sampling strategy to sample from returned frame-inds
        clip_centers = img_metas['frame_inds'][clip_length//2::clip_length] + video_start_idx
        ### get absolute segment's center-frame's indices
        clip_centers = clip_centers + int(timeframe_range[0])

        ##################### Extract Track-bboxes at clip centers and Get its roi-aligned features. #####################
        ### **NOTE: When employing TSN-style sampling strategy,
        ### This would fail if no track bboxes are found for any segment. And also for tracks shorter than 5 frames-length.
        ### This should work for missing segment-track-bboxes (assuming the remaining segments have good enough signal for explanation prediction)
        track_framnos = tcropped_track_bboxes[:, 4]
        no_missed_timeframes = 0
        ### for each track focus on only those frames, where clip_center feature is available, use the bbox at that frame to extract the features..
        for ii, ci in enumerate(clip_centers):
            if(ci < track_framnos[0] or ci > track_framnos[-1]):
                continue
            if(np.sum(track_framnos==ci)==0):
                no_missed_timeframes+=1
                # continue
                ### missing timeframe_bbox (interpolate)
                for ti in range(len(track_framnos)):
                    if(track_framnos[ti]-ci > 0):
                        break
                # print(track_framnos[ti], track_framnos[ti-1])
                w1_gap = abs(track_framnos[ti] - ci)
                w2_gap = abs(ci - track_framnos[ti-1])
                total_gap = w1_gap + w2_gap
                w1, w2 = (w1_gap/total_gap , w2_gap/total_gap)
                tracki_bbox = w1 * tcropped_track_bboxes[ti, 0:4][0] + w2 * tcropped_track_bboxes[ti-1, 0:4][0]
            else:
                tracki_bbox = tcropped_track_bboxes[track_framnos==ci, 0:4][0]
            tracki_bbox = tracki_bbox * bbox_xyxy_scaling

            ### ii is the clip index , the below roi corresponds to ii-th clip (so its required for roi-align to know from which clip to extract feat from)
            np_rois = np.array([[ii, tracki_bbox[0], tracki_bbox[1], tracki_bbox[2], tracki_bbox[3]]], dtype=np.float32)
            if(tracks_rois is None):
                tracks_rois = np_rois
            else:
                tracks_rois = np.vstack((tracks_rois, np_rois))
        if(tracks_rois is None and n_trys > 10):
            print("Tried many times, not getting matching centers:", n_trys, clip_centers, track_framnos)

    if(img_flip):
        tracks_rois_ = tracks_rois.copy()
        tracks_rois_[..., 1::4] = final_img_width - tracks_rois[..., 3::4]
        tracks_rois_[..., 3::4] = final_img_width - tracks_rois[..., 1::4]
        tracks_rois = tracks_rois_
    tracks_rois = torch.from_numpy(tracks_rois).to(device)

    roi_context_feats = ret_feats['cls_head.avg_pool'].squeeze().cpu().numpy()
    roi_feats = roi_align(ret_feats['backbone.layer4'], tracks_rois, (1, 1), 1/32, 0, 'avg', True).squeeze().cpu().numpy()
    if(roi_feats.shape[0]==2048 or roi_feats.shape[0]>1000):
        roi_feats = roi_feats[None, :]
        roi_context_feats = roi_context_feats[None, :]

    ### temporal max-pooling, or take consensus (avg. between segments)
    # roi_temporally_aggregated_feats = np.max(roi_feats,0)
    roi_temporally_aggregated_feats = np.mean(roi_feats,0)
    del ret_feats, results, img_metas
    # return roi_temporally_aggregated_feats, roi_feats, roi_context_feats, img_flip, tracks_rois   ### (2048), (N,2048), (N, 2048)
    return roi_temporally_aggregated_feats, roi_feats, roi_context_feats   ### (2048), (N,2048), (N, 2048)



############### Train pipeline (Class-balancing per mini-batch not done)
### for best accuracy
best_loss = 0
val_increase_count = 0
prev_loss = 0
weighted_loss = False

### for perScenario set perClip = False.
perClip = False
if(perClip):
    ### should be a multiple of the no. segments (used in sampling strategy, i.e., 8, check tsn_config.py file)
    batch_size = 32
    n_segments = 8
    print("Training a  perClip model.")
else:
    batch_size = 16
    n_segments = 1
    print("Training a  perScenario model.")
batch_samples_size = int(batch_size/n_segments)
print("No. of samples per mini-batch:", batch_samples_size)

log_dir = 'model_train_logs/'
os.makedirs(log_dir,exist_ok=True)
model_dir = 'saved_models/'
os.makedirs(model_dir,exist_ok=True)

### Sbkp contains class-specific dataset-indices list
### c_max_samples is the no. of samples to retrieve from each class (by over/under-sampling)
Sbkp, c_max_samples = prepLabelsDS(trainset_labels, num_samples=len(trainset_labels), batch_size = batch_samples_size)
print("c_max_samples:", c_max_samples)

if(perClip):
    model_name = 'perClipModel'
else:
    ### Models can be trained with IB: Instance-balanced sampling or CB: Class-balanced sampling
    
    ### this is with 2048 input (no Global Context, only TOI Aligned features)
    model_name = 'perScenarioModel_IB'
    
    ### this is with 4096 input (both Global Context and TOI Aligned features)
    model_name = 'perScenarioModel_wContextConcat_IB_interim128'  
    
    model_name += '_wextras'
if(weighted_loss == True):
    model_name += '_weightedCE'
print("Model name:", model_name)
print()

interim_layerdim = 128
if('Context' in model_name):
    ### eXplanation generator MLP
    model = nn.Sequential(
        nn.Linear(4096, interim_layerdim),
        nn.ReLU(),
        nn.Linear(interim_layerdim, num_classes)
    )
else:
    ### eXplanation generator MLP
    model = nn.Sequential(
        nn.Linear(2048, interim_layerdim),
        nn.ReLU(),
        nn.Linear(interim_layerdim, num_classes)
    )
model = model.to(device)

### do weighted cross-entropy based on the labels distribution
if(weighted_loss == False):
    loss_fn = nn.CrossEntropyLoss()
else:
    ### for weighted cross-entropy
    label_weights = np.array(1- trainset_labels_dist['counts']/trainset_labels_dist['counts'].sum())
    label_weights = torch.from_numpy(label_weights.astype('float32')).to(device)
    print("CE label weights in order:", label_weights)
    loss_fn = nn.CrossEntropyLoss(label_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

if('_IB' in model_name):
    IB_flag = True
else:
    IB_flag = False

num_epochs = 50
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corr_exp1 = 0
        n_examples = 0
        
        if phase == 'train':
            model.train()
            idxs_order = list(range(len(trainset['video_dir'])))
            if(IB_flag):
                np.random.shuffle(idxs_order)
            else:
                ### below function requires the labels to be in the original order (so shuffling is commented)
                epoch_batches = getClassBalancedBatches(Sbkp, c_max_samples, batch_size = batch_samples_size)
                
            all_video_dirs = [trainset['video_dir'][idi] for idi in idxs_order]
            all_track_bboxes = [trainset['merged_bboxes'][idi] for idi in idxs_order]
            all_labels = trainset_labels[idxs_order]
            if(IB_flag):
                #### when class-balancing is not required (uncomment shuffling line, and comment epoch_batches line above, and uncomment below)
                epoch_batches = []
                for batch_idx in range(0, len(all_labels), batch_samples_size):
                    epoch_batches.append(list(range(batch_idx, min(batch_idx+batch_samples_size, len(all_labels)))))
        else:
            model.eval()
            idxs_order = list(range(len(valset['video_dir'])))
            all_video_dirs = [valset['video_dir'][idi] for idi in idxs_order]
            all_track_bboxes = [valset['merged_bboxes'][idi] for idi in idxs_order]
            all_labels = valset_labels[idxs_order]
            ### no shuffling to maintain labels order
            epoch_batches = []
            for batch_idx in range(0, len(all_labels), batch_samples_size):
                epoch_batches.append(list(range(batch_idx, min(batch_idx+batch_samples_size, len(all_labels)))))
        if(len(epoch_batches[-1])<=1):
            epoch_batches = epoch_batches[:-1]
        num_batches_ = len(epoch_batches) #len(all_labels) / batch_samples_size
        print("Total no. of batches:",num_batches_)
        
        for batch_idx, ebi in tqdm(enumerate(epoch_batches)):
            ### extract indices from each prepared mini-batch
            video_dirs_batch = [all_video_dirs[bii] for bii in ebi]
            track_bboxes_batch = [all_track_bboxes[bii] for bii in ebi]
            labels_batch = [all_labels[bii] for bii in ebi]
            # print(labels_batch)
        # for batch_idx in range(0, len(all_labels), batch_samples_size):
        #     video_dirs_batch = all_video_dirs[batch_idx:batch_idx+batch_samples_size]
        #     track_bboxes_batch = all_track_bboxes[batch_idx:batch_idx+batch_samples_size]
        #     labels_batch = all_labels[batch_idx:batch_idx+batch_samples_size]
            
            batch_feats = None
            batch_labels = []
            for video_dir, track_bboxes, labels_ in zip(video_dirs_batch, track_bboxes_batch, labels_batch):
                if(phase=='train'):
                    # ### Note: temporal range should actually be defined for frame-nos. but for sparse temporal data, no. of frames in track is considered.
                    # ### min temp. range, 14 coz it should be atleast greater than 8_segments+5_cliplen = 13 frames. 
                    # ### This maybe cropping out the original signal which maybe located in a very small temporal duration somewhere midway
                    # temporal_window_range_ = (min(14/len(track_bboxes), 1.0), 1.0)
                    # roi_TAF, roi_F, roi_CF = get_ar_feats(video_dir, track_bboxes, phase, temporal_window_range = temporal_window_range_)
                    roi_TAF, roi_F, roi_CF = get_ar_feats(video_dir, track_bboxes, phase)
                else:
                    roi_TAF, roi_F, roi_CF = get_ar_feats(video_dir, track_bboxes, phase, temporal_window_range = (1.0, 1.0))
                if(batch_feats is None):
                    if(perClip):
                        batch_feats = roi_F
                    else:
                        if('Context' not in model_name):
                            batch_feats = roi_TAF
                        else:
                            # ### adding temporally aggregated context feats.
                            # batch_feats = roi_TAF + np.mean(roi_CF,0)
                            ### concatenating temporally aggregated context feats. (-1 for last axis)
                            batch_feats = np.concatenate((roi_TAF , np.mean(roi_CF,0)), -1)
                else:
                    if(perClip):
                        batch_feats = np.vstack((batch_feats, roi_F))
                    else:
                        if('Context' not in model_name):
                            batch_feats = np.vstack((batch_feats, roi_TAF))
                        else:
                            # ### adding temporally aggregated context feats.
                            # batch_feats = np.vstack((batch_feats, roi_TAF + np.mean(roi_CF,0)))
                            ### concatenating temporally aggregated context feats. (-1 for last axis)
                            batch_feats = np.vstack((batch_feats, np.concatenate((roi_TAF , np.mean(roi_CF,0)), -1) ))
                if(perClip):
                    batch_labels.extend( [labels_]*roi_F.shape[0] )
                    n_examples += roi_F.shape[0]
                else:
                    batch_labels.extend( [labels_] )
                    n_examples += 1
            
            batch_labels = np.array(batch_labels)
            if(perClip):
                idxs_order = list(range(len(batch_labels)))
                np.random.shuffle(idxs_order)
                batch_labels = batch_labels[idxs_order]
                batch_feats = batch_feats[:, idxs_order,:]
            
            batch_feats = torch.from_numpy(batch_feats).to(device)
            batch_labels = torch.from_numpy(batch_labels).to(device)
            # print(batch_feats.shape, batch_labels.shape)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                output = model(batch_feats)
                _, pred_exp1 = torch.max(output, 1)
                loss = loss_fn(output, batch_labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_corr_exp1 += (batch_labels.cpu() == pred_exp1.cpu()).sum()
            
            # Print the average loss in a mini-batch.
            if batch_idx % 10 == 0:
                print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                      .format(phase.upper(), epoch+1, num_epochs, batch_idx, num_batches_, loss.item()))
            del batch_feats, batch_labels, loss, _, pred_exp1, output, roi_TAF, roi_F, roi_CF

        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / num_batches_
        epoch_acc_exp1 = running_corr_exp1.double() / n_examples
        
        print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f} \n'
              .format(phase.upper(), epoch+1, num_epochs, epoch_loss, epoch_acc_exp1))
        
        # Log the loss and accuracy in an epoch.
        with open(os.path.join(log_dir, '{}-{}-log-epoch-{:02}.txt')
                  .format(model_name, phase, epoch+1), 'w') as f:
            f.write(str(epoch+1) + '\t'
                    + str(epoch_loss) + '\t'
                    + str(epoch_acc_exp1.item()) + '\t'
                    + str(0))

        if phase == 'val':
            # if epoch_loss < best_loss:
            if epoch_acc_exp1 > best_loss:
                # print("At epoch:",epoch+1,"best loss from:\t",best_loss, "\tto\t",epoch_loss)
                print("At epoch:",epoch+1,"best acc. from:\t",best_loss, "\tto\t",epoch_acc_exp1)
                # best_loss = epoch_loss
                best_loss = epoch_acc_exp1
                # torch.save(model, os.path.join(model_dir, '{}-best_model.pt'.format(model_name)))
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(model_dir, '{}-e{}-best_model.ckpt'.format(model_name, epoch+1)))
            # if epoch_loss > prev_loss:
            if epoch_acc_exp1 < prev_loss:
                val_increase_count += 1
            else:
                val_increase_count = 0
            # prev_loss = epoch_loss
            prev_loss = epoch_acc_exp1

