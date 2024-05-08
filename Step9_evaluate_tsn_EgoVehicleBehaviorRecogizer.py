######### Step9_evaluate_tsn_egobehavrecog.py
import time
from glob import glob
import numpy as np
import os, json
import pandas as pd
from tqdm import tqdm

### get y_true and y_pred labels
def get_classwise_accuracy(gt_file_path, pred_file_path):
    labels_mapping = {'0':'slowdown', '1':'deviate', '2':'turn_and_slowdown'}
    labels_totals = {'slowdown': 0, 'deviate': 0, 'turn_and_slowdown': 0}
    labels_matched = {'slowdown': 0, 'deviate': 0, 'turn_and_slowdown': 0}

    labels_acc = {}
    f = open(pred_file_path)
    data = json.load(f)
    itr=-1
    with open(gt_file_path) as f:
        for line in f:
            itr+=1
            gt_filedir = line.strip().split(' ')[0]
            gt_labels = line.strip().split(' ')[2:]
            ##### total buggy line below fucker
            # top_k_predlabels = np.where(np.argsort(data[itr])>=(len(labels_mapping.keys())-len(gt_labels)))[0]
            top_k_predlabels = np.argsort(data[itr])[(len(labels_mapping.keys())-len(gt_labels)):]
            for labeli in gt_labels:
                labels_totals[labels_mapping[labeli]] += 1
                if(int(labeli) in top_k_predlabels):
                    labels_matched[labels_mapping[labeli]] += 1
                    
    for lk in labels_matched.keys():
        labels_acc[lk] = round((labels_matched[lk]/labels_totals[lk])*100,1)        
    return labels_acc, labels_matched, labels_totals

#### small ds wrt yield nos
di = 'work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_iddx_front_IB/'
#di = 'work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_iddx_rear_IB/'
#di = 'work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_rgb_iddx_front_IB/'
#di = 'work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_rgb_iddx_rear_IB/'

# python mmaction2/tools/train.py work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_iddx_front_IB/tsn_config-activitynet-rgb_iddx_latest_wCorrectCropAug.py

###### gt file path
gt_file_path = 'iddx_clips_rawframe_front/val_flow/annotations.txt'
#gt_file_path = 'iddx_clips_rawframe_rear/val_flow/annotations.txt'
#gt_file_path = 'iddx_clips_rawframe_front/val/annotations.txt'
#gt_file_path = 'iddx_clips_rawframe_rear/val/annotations.txt'

os.makedirs(di,exist_ok=True)
# while(True):
#     #time.sleep(100)
    
allmodels = np.array([di+'epoch_'+str(epnum)+'.pth' for epnum in range(1,41)])
# allmodels = np.array(glob(di+"epoch*"))
allmodels_epochs = np.array([int(mi.split('epoch_')[1][:-4]) for mi in allmodels])
epoch_order = np.argsort(allmodels_epochs)

exp_df = {'epoch':[], 'slowdown': [], 'deviate': [], 'turn_and_slowdown': [], 'AvgAcc':[], 'top1_acc':[], 'avg_acc':[]}
allmodels_epochs = allmodels_epochs[epoch_order]
allmodels = allmodels[epoch_order]

for ei, mi in enumerate(allmodels):
    ### check if model exists
    if(not os.path.exists(mi)):
        continue
    ### check if model's val-perf. json exists
    pred_file_path = di+'temp'+str(allmodels_epochs[ei])+'.json'
    if(os.path.exists(pred_file_path)):
        continue

    print(ei,mi)
    
    testout = os.popen('python mmaction2/tools/test.py '+di+'tsn_config-activitynet-rgb_iddx_latest_wCorrectCropAug.py '+mi+' --eval top_k_accuracy mean_class_accuracy --out '+di+'temp'+str(allmodels_epochs[ei])+'.json').read()

    exp_df['top1_acc'].append(float(testout.split('top1_acc: ')[1].strip().split('\n')[0]))
    exp_df['avg_acc'].append(float(testout.split('mean_class_accuracy: ')[1].strip().split('\n')[0]))
    exp_df['epoch'].append(allmodels_epochs[ei])

    labels_accuracy, labels_gtpred_matched, labels_gt_totals = get_classwise_accuracy(gt_file_path, pred_file_path)

    avgacc = 0
    for lic in labels_accuracy.keys():
        exp_df[lic].append(labels_accuracy[lic])
        avgacc+=labels_accuracy[lic]
    avgacc/=len(labels_accuracy.keys())
    exp_df['AvgAcc'].append(avgacc)
    # print("avg acc, wavg acc :", avgacc, exp_df['top1_acc'][-1])
    print(exp_df)

exp_df = pd.DataFrame(exp_df)
if(os.path.exists(di+'all_epoch_models_accuracies.csv')):
    exp_df_ = pd.read_csv(di+'all_epoch_models_accuracies.csv')
    exp_df = pd.concat([exp_df_,exp_df])
exp_df.to_csv(di+'all_epoch_models_accuracies.csv',index=False)
# exp_df
