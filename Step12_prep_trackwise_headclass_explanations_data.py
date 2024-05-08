######### Step12_prep_trackwise_headclass_explanations_data.py    (ignoring tail classes for explanation prediction)
### Note: Risk-objects referenced in the code is synonymous with Important Objects

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from mmaction.apis import init_recognizer, inference_recognizer
from mmcv.ops import roi_align
from collections import Counter


dstypes = ['train', 'val']
for dstype in dstypes:
    # dstype = 'train'
    gt_file_path = 'iddx_clips_rawframe_front/'+dstype+'_flow/annotations.txt'

    ### output file
    pertrack_data_path = 'pertrack_'+dstype+'_data_atpmerged_dict_wextras.npy'
    pertrack_data = {'video_dir':[], 'merged_bboxes':[], 'explanation_class':[], 'object_gt_class':[]}

    remove_labels = ['Other', 'Avoid Stopped Vehicle', 'Deviate', 'Avoid Obstruction', 'Merging', 'Red Light', 'Right Turn', 'U-Turn', 'Left Turn', 'Slowing Down']
    # remove_labels += ["on-road vehicle", "avoid on-road vehicle", "lane-change", "moving on merging road", "red light", "avoid obstruction", "obstruction", "right-turn", "u-turn", "left-turn"]

    f = open(gt_file_path)
    for line in tqdm(f):
        video = line.strip().split(' ')[0]

        trackwise_data_path = 'iddd_trackwise_'+dstype+'_data/'+video.split('/')[-1]+'_atpmerged_dict.npy'
        fused_trackdata = np.load(trackwise_data_path, allow_pickle=True).tolist()
        ntracks = len(fused_trackdata['merged_bboxes'])

        if(ntracks>0):
            for tii in range(ntracks):
                explanation_class = fused_trackdata['atp_explanation'][tii]
                object_class = fused_trackdata['atp_class'][tii]
                # print(explanation_class, object_class)

                ### remove_labels
                if(explanation_class not in remove_labels):
                    pertrack_data['explanation_class'].append(explanation_class)
                    pertrack_data['video_dir'].append(video.split('/')[-1])
                    pertrack_data['merged_bboxes'].append(fused_trackdata['merged_bboxes'][tii])
                    pertrack_data['object_gt_class'].append(object_class)

    print("Total tracks in "+dstype+":",len(pertrack_data['explanation_class']))
    np.save(pertrack_data_path, pertrack_data)

    label_counts = dict(Counter(pertrack_data['explanation_class']))
    lidxs = np.argsort(list(label_counts.values()))[::-1]
    labels_df = pd.DataFrame({'labels':np.array(list(label_counts.keys()))[lidxs], 'counts':np.array(list(label_counts.values()))[lidxs]})
    labels_df.to_csv('riskobj_explanation_labels_dist_'+dstype+'_wextras.csv',index=False)
    print(labels_df)