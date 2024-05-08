######### Step5_create_MOTdata_with_GT_IOtracks.py
### Note: Risk-objects referenced in the code is synonymous with Important Objects
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, copy, random, cv2
from glob import glob
from tqdm import tqdm
from collections import Counter

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
# mot_classes = np.array(list(pd.read_csv('mot_classes.txt', header=None)[0]))

### gathering MOT data
dstypes = ['train', 'val']
for dstype in dstypes:
    # dstype = 'train'
    # dstype = 'val'
    mottrack_npys = glob('iddd_trackwise_'+dstype+'_data/*.npy')
    mottrack_npys_ = []
    for ni in mottrack_npys:
        if('_atpmerged_dict' in ni):
            continue
        mottrack_npys_.append(ni)
    mottrack_npys = mottrack_npys_
    mottrack_csvs = [ni.replace('_dict.npy','_nobox_df.csv') for ni in mottrack_npys]
    eventids  = [int(ni.split('/')[-1].split('_')[0]) for ni in mottrack_npys]
    print(f"Total no. events in {dstype} set:", len(eventids))

    pertrack_data_path = 'pertrack_'+dstype+'_data_motwise_dict.npy'
    pertrack_data = {'video_dir':[], 'track_bboxes':[], 'riskobj_class':[], 'explanation_class':[], 'motobj_class':[], 'motobj_label':[], 'gtobj_class':[], 'mot_trackid':[], 'atp_trackid':[]}
    for ei, ri in tqdm(enumerate(eventids)):
        trackdata_csv_path = mottrack_csvs[ei]
        trackdata_csv_df = pd.read_csv(trackdata_csv_path)

        trackdata_path = mottrack_npys[ei]
        trdf = np.load(trackdata_path, allow_pickle=True)
        trdf = trdf.tolist()

        video_dir = trackdata_path.replace('_dict.npy','').split('/')[-1]
        ref_start = int(video_dir.split('_')[-1].split('to')[0])

        for moti in range(trackdata_csv_df.shape[0]):
            ### excluding which are not road-objects
            if(trackdata_csv_df['is_mot_object_class_valid'][moti]==0):
                continue
            all_bboxes = np.vstack(trdf['bboxes'][moti])
            relative_bbox_framenos = all_bboxes[:,-1]
            absolute_video_bbox_framenos = relative_bbox_framenos + ref_start
            all_bboxes[:,-1] = absolute_video_bbox_framenos

            pertrack_data['video_dir'].append(video_dir)
            pertrack_data['track_bboxes'].append(all_bboxes)
            pertrack_data['riskobj_class'].append(trackdata_csv_df['is_mot_risk_object'][moti])
            pertrack_data['explanation_class'].append(trackdata_csv_df['mot_risk_object_atpX_label'][moti])
            pertrack_data['motobj_class'].append(mot_classes[trackdata_csv_df['label'][moti]])
            pertrack_data['gtobj_class'].append(trackdata_csv_df['mot_risk_object_atpObj_label'][moti])
            pertrack_data['motobj_label'].append(trackdata_csv_df['label'][moti])
            pertrack_data['mot_trackid'].append(trackdata_csv_df['trackid'][moti])
            pertrack_data['atp_trackid'].append(trackdata_csv_df['mot_risk_object_atptrackid'][moti])

    print("Total tracks in "+dstype+":",len(pertrack_data['riskobj_class']))
    np.save(pertrack_data_path, pertrack_data)

    label_counts = dict(Counter(pertrack_data['riskobj_class']))
    lidxs = np.argsort(list(label_counts.values()))[::-1]
    labels_df = pd.DataFrame({'labels':np.array(list(label_counts.keys()))[lidxs], 'counts':np.array(list(label_counts.values()))[lidxs]})
    labels_df.to_csv('riskobj_labels_MOTwise_dist_'+dstype+'.csv',index=False)
    print(labels_df)
    print("*Here label 1 is for Important Object\n\n")
    
