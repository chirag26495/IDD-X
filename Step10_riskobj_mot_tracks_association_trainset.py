######### Step10_riskobj_mot_tracks_association_trainset.py  (to improve the temporal consistency of Risk Object annotations)
### Note: Risk-objects referenced in the code is synonymous with Important Objects

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, random, cv2, json
from glob import glob
from tqdm import tqdm

def iou_and_ubox(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    ubox = [xA, yA, xB, yB]
    return iou, ubox

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

### replace below path to downloaded data folder
relative_src_path = "./"  # "../IDDX/"

### annotation json read
with open(relative_src_path+"iddx_annotations.json", "r") as outfile:
	iddx_annos_json = outfile.read()
iddx_annos_json = json.loads(iddx_annos_json)
alljson_eventids = []
for ei in iddx_annos_json:
    alljson_eventids.append(ei['event_id'])

iou_thresh = 0.2
root_dir = 'IDD-D_Yolo-v4-Model/'
namesfile = root_dir + 'idd.names'
mot_classes = load_class_names(namesfile)
mot_classes = np.array(mot_classes)

dstypes = ['train', 'val']
for dstype in dstypes:
    # dstype = 'train'
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

    for ei, ri in tqdm(enumerate(eventids)):
        trackdata_csv_path = mottrack_csvs[ei]
        trackdata_csv_df = pd.read_csv(trackdata_csv_path)

        trackdata_path = mottrack_npys[ei]
        trdf = np.load(trackdata_path, allow_pickle=True)
        trdf = trdf.tolist()

        ### iterate over each primary eventid json_ri
        json_ri = alljson_eventids.index(ri)
        ref_start, ref_end = iddx_annos_json[json_ri]['start_frame'], iddx_annos_json[json_ri]['end_frame']

        is_mottrack_risk_object = [0]*trackdata_csv_df.shape[0]
        risk_object_mot2_match = [-1]*trackdata_csv_df.shape[0]
        risk_object_mot2_match_maxiou = [0]*trackdata_csv_df.shape[0]
        risk_object_trackid = [-1]*trackdata_csv_df.shape[0]
        risk_object_explanation = ['']*trackdata_csv_df.shape[0]
        risk_object_gt_class = ['']*trackdata_csv_df.shape[0]
        risk_object_mot_class = ['']*trackdata_csv_df.shape[0]
        mottrack_risk_object_maxiou = [0]*trackdata_csv_df.shape[0]
        is_mottrack_main_risk_object = [0]*trackdata_csv_df.shape[0]
        no_of_missing_motframes = [0]*trackdata_csv_df.shape[0]
        motframes_timegap_max = [0]*trackdata_csv_df.shape[0]
        motframes_timegap_mean = [0]*trackdata_csv_df.shape[0]
        motframes_timegap_std = [0]*trackdata_csv_df.shape[0]
        # print(trackdata_csv_df.shape)

        for atp_idx, atpi in enumerate(iddx_annos_json[json_ri]['IOs']):
            atp_io_tracks_wframenum = np.array(atpi['IO_track_and_frameno'])

            cur_start, cur_end = atp_io_tracks_wframenum[0, -1], atp_io_tracks_wframenum[-1, -1]
            if(cur_start < ref_start):
                cur_start = ref_start
            if(cur_end > ref_end):
                cur_end = ref_end

            ### objects coming from rear to front view are also considered below---->  (filter them out those scenes before hand)
            atp_track_id, atp_explanation, atp_obj_class = atpi['IO_id'], atpi['IO_explanation'], atpi['IO_category']

            atp_mot_frame_maxious = []
            atp_mot_frame_maxious_motis = []
            atp_mot_frame_maxious2 = []
            atp_mot_frame_maxious_motis2 = []
            for video_frame_no in range(cur_start, cur_end+1):
                atp_bbox = atp_io_tracks_wframenum[atp_io_tracks_wframenum[:, -1]==video_frame_no, :4]
                if(len(atp_bbox)>0):
                    atp_bbox = atp_bbox[0,:]
                    xmin, ymin, xmax, ymax = atp_bbox
                    ### find if it is in front or rear view (using mid-point of ymin and ymax)
                    ymid = (ymin+ymax)/2
                    if(ymid>2160/2):
                        ## in-rear
                        continue
                    xmin = max(0, xmin)
                    xmax = min(1920-1, xmax)
                    ymin = max(0, ymin)
                    ymax = min(2160//2-1, ymax)
                    atp_bbox = np.array([xmin, ymin, xmax, ymax])

                    relative_frameno = video_frame_no-ref_start
                    max_iou = 0
                    max_iou_moti = -1
                    max_iou2 = 0
                    max_iou_moti2 = -1
                    for moti in range(trackdata_csv_df.shape[0]):
                        if(is_mottrack_risk_object[moti]==1):
                            continue
                        all_bboxes = np.vstack(trdf['bboxes'][moti])
                        bbox_framenos = all_bboxes[:,-1]
                        mot_boxi = np.where(bbox_framenos==relative_frameno)[0]
                        if(len(mot_boxi)==0):
                            continue
                        mot_box = all_bboxes[mot_boxi[0],:4]

                        motatp_iou, _ = iou_and_ubox(mot_box, atp_bbox)
                        if(max_iou < motatp_iou):
                            max_iou = motatp_iou
                            max_iou_moti = moti
                        if(max_iou2 < motatp_iou and max_iou > motatp_iou):
                            max_iou2 = motatp_iou
                            max_iou_moti2 = moti
                    atp_mot_frame_maxious.append(max_iou)
                    atp_mot_frame_maxious_motis.append(max_iou_moti)
                    atp_mot_frame_maxious2.append(max_iou2)
                    atp_mot_frame_maxious_motis2.append(max_iou_moti2)

            if(len(atp_mot_frame_maxious)==0):
                atp_matched_moti = -1
                # print('No match. atp object in rear view. atp object class: ', atp_obj_class, atp_bbox, ymid)
            elif(np.max(atp_mot_frame_maxious) > iou_thresh):
                atp_matched_moti = atp_mot_frame_maxious_motis[np.argmax(atp_mot_frame_maxious)]
                is_mottrack_risk_object[atp_matched_moti] = 1
                risk_object_trackid[atp_matched_moti] = atp_track_id

                risk_object_explanation[atp_matched_moti] = atp_explanation
                risk_object_gt_class[atp_matched_moti] = atp_obj_class
                risk_object_mot_class[atp_matched_moti] = mot_classes[trdf['label'][atp_matched_moti]]
                mottrack_risk_object_maxiou[atp_matched_moti] = np.max(atp_mot_frame_maxious)
                if(atpi==ri):
                    is_mottrack_main_risk_object[atp_matched_moti] = 1

                all_bboxes = np.vstack(trdf['bboxes'][atp_matched_moti])
                bbox_framenos = all_bboxes[:,-1]
                timeframes_length = len(bbox_framenos)
                actual_length = trdf['end_frameno'][atp_matched_moti] - trdf['start_frameno'][atp_matched_moti] + 1
                no_of_missing_motframes[atp_matched_moti] = actual_length - timeframes_length
                dts = []
                for fi in range(1, timeframes_length):
                    dts.append(bbox_framenos[fi] - bbox_framenos[fi-1])
                if(len(dts)>0):
                    motframes_timegap_max[atp_matched_moti] = np.max(dts)
                    motframes_timegap_mean[atp_matched_moti] = np.round(np.mean(dts),1)
                    motframes_timegap_std[atp_matched_moti] = np.round(np.std(dts),1)

                atp_matched_moti2 = atp_mot_frame_maxious_motis2[np.argmax(atp_mot_frame_maxious2)]
                risk_object_mot2_match[atp_matched_moti] = atp_matched_moti2
                risk_object_mot2_match_maxiou[atp_matched_moti] = np.max(atp_mot_frame_maxious2)
                # print(atp_obj_class, atp_matched_moti, mot_classes[trdf['label'][atp_matched_moti]], (cur_start-ref_start), (cur_end-ref_start), trdf['start_frameno'][atp_matched_moti], trdf['end_frameno'][atp_matched_moti], np.max(atp_mot_frame_maxious))
            else:
                atp_matched_moti = -1
                # print('No matching mot object track found. atp object class: ', atp_obj_class)

        trackdata_csv_df['is_risk_object'] = is_mottrack_risk_object
        trackdata_csv_df['risk_object_trackid'] = risk_object_trackid
        trackdata_csv_df['risk_object_explanation'] = risk_object_explanation
        trackdata_csv_df['risk_object_gt_class'] = risk_object_gt_class
        trackdata_csv_df['risk_object_mot_class'] = risk_object_mot_class
        trackdata_csv_df['mottrack_risk_object_maxiou'] = mottrack_risk_object_maxiou
        trackdata_csv_df['is_mottrack_main_risk_object'] = is_mottrack_main_risk_object
        trackdata_csv_df['no_of_missing_motframes'] = no_of_missing_motframes
        trackdata_csv_df['motframes_timegap_max'] = motframes_timegap_max
        trackdata_csv_df['motframes_timegap_mean'] = motframes_timegap_mean
        trackdata_csv_df['motframes_timegap_std'] = motframes_timegap_std
        trackdata_csv_df['risk_object_mot2_match'] = risk_object_mot2_match
        trackdata_csv_df['risk_object_mot2_match_maxiou'] = risk_object_mot2_match_maxiou
        trackdata_csv_df.to_csv(trackdata_csv_path, index=False)    
        # np.save(trackdata_path, trdf)
