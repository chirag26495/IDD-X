######### Step4_associate_MOTdata_with_GT_IOtracks.py
### Note: Risk-objects referenced in the code is synonymous with Important Objects
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, copy, random, cv2, json
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

root_dir = 'IDD-D_Yolo-v4-Model/'
namesfile = root_dir + 'idd.names'
mot_classes = load_class_names(namesfile)
# mot_classes = np.array(list(pd.read_csv('mot_classes.txt', header=None)[0]))

# ### shortlisted valid risk-objects classes in MOT [NOT REQUIRED BELOW since iddd labels are used]
# road_object_classes = ['truck', 'baby', 'trailer_truck', 'motorcycle', 'car_(automobile)', 'bicycle', 'bus_(vehicle)', 'train_(railroad_vehicle)', 'goat', 'baby_buggy', 
#  'tractor_(farm_equipment)', 'cow', 'pickup_truck', 'motor_scooter', 'cab_(taxi)', 'horse', 'dog', 'minivan', 'cart', 'handcart','pickup_truck', 
#  'school_bus', 'sheep', 'tow_truck', 'jeep', 'bull', 'calf']

### gathering MOT data
dstypes = ['train', 'val']
for dstype in dstypes:
    mottrack_npys = glob('iddd_trackwise_'+dstype+'_data/*.npy')
    # mottrack_npys = glob('iddd_trackwise_val_data/*.npy')
    # print(len(mottrack_npys), len(mottrack_npys)/2)
    mottrack_npys_ = []
    for ni in mottrack_npys:
        if('_atpmerged_dict' in ni):
            continue
        mottrack_npys_.append(ni)
    mottrack_npys = mottrack_npys_

    mottrack_csvs = [ni.replace('_dict.npy','_nobox_df.csv') for ni in mottrack_npys]
    eventids  = [int(ni.split('/')[-1].split('_')[0]) for ni in mottrack_npys]
    print(f"Total no. events in {dstype} set:", len(eventids))
    
    ### for each driving scenario we have extracted the MOT data in which each object is labelled as important/not-important using IOU with GT IO tracks
    for ei, ri in tqdm(enumerate(eventids)):
        trackdata_csv_path = mottrack_csvs[ei]
        trackdata_csv_df = pd.read_csv(trackdata_csv_path)

        trackdata_path = mottrack_npys[ei]
        trdf = np.load(trackdata_path, allow_pickle=True)
        trdf = trdf.tolist()

        ### iterate over each primary eventid json_ri
        json_ri = alljson_eventids.index(ri)
        ref_start, ref_end = iddx_annos_json[json_ri]['start_frame'], iddx_annos_json[json_ri]['end_frame']

        ### min_stIOU is sufficient for identifying the risk-object, while median_iou is to identify which atp does it prominently belong to.
        min_stIOU_mots_atps = np.ones((trackdata_csv_df.shape[0], len(iddx_annos_json[json_ri]['IOs']))) * -1
        median_stIOU_mots_atps = np.ones((trackdata_csv_df.shape[0], len(iddx_annos_json[json_ri]['IOs']))) * -1
        max_stIOU_mots_atps = np.ones((trackdata_csv_df.shape[0], len(iddx_annos_json[json_ri]['IOs']))) * -1

        for atp_idx, atpi in enumerate(iddx_annos_json[json_ri]['IOs']):
            atp_io_tracks_wframenum = np.array(atpi['IO_track_and_frameno'])

            cur_start, cur_end = atp_io_tracks_wframenum[0, -1], atp_io_tracks_wframenum[-1, -1]
            if(cur_start < ref_start):
                cur_start = ref_start
            if(cur_end > ref_end):
                cur_end = ref_end

            ### objects coming from rear to front view are also considered below---->  (filter them out those scenes before hand)
            atp_track_id, atp_explanation, atp_obj_class = atpi['IO_id'], atpi['IO_explanation'], atpi['IO_category']

            ### empty array list init
            stIOUs_mots_atpi = []
            for moti in range(trackdata_csv_df.shape[0]):
                stIOUs_mots_atpi.append([])

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
                    for moti in range(trackdata_csv_df.shape[0]):
                        all_bboxes = np.vstack(trdf['bboxes'][moti])
                        mot_boxi = np.where(all_bboxes[:,-1]==relative_frameno)[0]
                        if(len(mot_boxi)==0):
                            continue
                        mot_box = all_bboxes[mot_boxi[0],:4]
                        motatp_iou, _ = iou_and_ubox(mot_box, atp_bbox)
                        stIOUs_mots_atpi[moti].append(motatp_iou)

            ### temporal aggregation of ious for each moti w.r.t. current atp
            for moti in range(trackdata_csv_df.shape[0]):
                if(len(stIOUs_mots_atpi[moti])>0):
                    min_stIOU_mots_atps[moti, atp_idx] = np.min(stIOUs_mots_atpi[moti])
                    median_stIOU_mots_atps[moti, atp_idx] = np.median(stIOUs_mots_atpi[moti])
                    max_stIOU_mots_atps[moti, atp_idx] = np.max(stIOUs_mots_atpi[moti])

        ############# spatio-temporal IOU between gt_risk-obj_tracks and mot_obj_tracks for labelling mot_objs as risk-obj.
        min_stIOU_threshold = 0.2
        median_stIOU_threshold = 0.3
        is_mot_risk_object = [0] * trackdata_csv_df.shape[0]
        mot_risk_object_atptrackid = [-1] * trackdata_csv_df.shape[0]
        mot_risk_object_atpX_label = [''] * trackdata_csv_df.shape[0]
        mot_risk_object_atpObj_label = [''] * trackdata_csv_df.shape[0]
        mot_risk_object_atpIOU = [0] * trackdata_csv_df.shape[0]
        for moti in range(trackdata_csv_df.shape[0]):
            atp_idxs = np.where(min_stIOU_mots_atps[moti, :] > min_stIOU_threshold)[0]
            if(len(atp_idxs) > 0):
                median_stIOUs_ = median_stIOU_mots_atps[moti, atp_idxs ]
                max_median_stIOUs_ = np.max(median_stIOUs_)
                if(max_median_stIOUs_ < median_stIOU_threshold):
                    continue
                moti_atp_idx = atp_idxs[ np.argmax(median_stIOUs_) ]
                atp_track_id, atp_explanation, atp_obj_class = atpi['IO_id'], atpi['IO_explanation'], atpi['IO_category']
                is_mot_risk_object[moti] = 1
                mot_risk_object_atpIOU[moti] = max_median_stIOUs_
                mot_risk_object_atptrackid[moti] = atp_track_id
                mot_risk_object_atpX_label[moti] = atp_explanation
                mot_risk_object_atpObj_label[moti] = atp_obj_class

        ############# marking mot object classes which are road objects
        is_mot_object_class_valid = [0] * trackdata_csv_df.shape[0]
        for moti, li in enumerate(trackdata_csv_df['label']):
            ### including all object classes
            if(True): ##if(mot_classes[li] in road_object_classes):
                is_mot_object_class_valid[moti] = 1

        trackdata_csv_df['is_mot_object_class_valid'] = is_mot_object_class_valid
        trackdata_csv_df['is_mot_risk_object'] = is_mot_risk_object
        trackdata_csv_df['mot_risk_object_atpIOU'] = mot_risk_object_atpIOU
        trackdata_csv_df['mot_risk_object_atptrackid'] = mot_risk_object_atptrackid
        trackdata_csv_df['mot_risk_object_atpX_label'] = mot_risk_object_atpX_label
        trackdata_csv_df['mot_risk_object_atpObj_label'] = mot_risk_object_atpObj_label
        trackdata_csv_df.to_csv(trackdata_csv_path, index=False)

    print('done!')
