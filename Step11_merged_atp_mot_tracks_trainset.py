######### Step11_merged_atp_mot_tracks_trainset.py  (merging Risk Object and their associated MOT tracks)
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

        risk_object_trackid = trackdata_csv_df['risk_object_trackid']
        risk_object_trackid = np.array(risk_object_trackid)

        trackwise_merged_data = {'atp_trackid':[], 'mot_id':[], 'is_atp_main_risk_object':[], 'merged_start_frameno':[], 'merged_end_frameno':[], 
                                 'no_of_missing_mergedframes':[], 'mergedframes_timegap_max':[], 'mergedframes_timegap_mean':[], 'mergedframes_timegap_std':[], 
                                 'merged_bboxes':[], 'atp_explanation':[], 'atp_class':[], 'mot_class':[], 'atp_mot_maxiou':[]}
        for atp_idx, atpi in enumerate(iddx_annos_json[json_ri]['IOs']):
            atp_io_tracks_wframenum = np.array(atpi['IO_track_and_frameno'])

            cur_start, cur_end = atp_io_tracks_wframenum[0, -1], atp_io_tracks_wframenum[-1, -1]
            if(cur_start < ref_start):
                cur_start = ref_start
            if(cur_end > ref_end):
                cur_end = ref_end

            ### objects coming from rear to front view are also considered below---->  (filter them out those scenes before hand)
            atp_track_id, atp_explanation, atp_obj_class = atpi['IO_id'], atpi['IO_explanation'], atpi['IO_category']

            ### preparing dataset only for mot-matching objects
            if(atp_track_id not in risk_object_trackid):
                continue

            moti = np.where(risk_object_trackid==atp_track_id)[0][0]
            mot_startframeno, mot_endframeno, motobj_class, motobj_iou = (trdf['start_frameno'][moti], trdf['end_frameno'][moti], trackdata_csv_df['risk_object_mot_class'][moti], trackdata_csv_df['mottrack_risk_object_maxiou'][moti])

            #### NOTE: all framnos below (including MOT) are in the same frame of reference: (ref_start, ref_end)  ---> corresponding to the ego-action-event
            atp_startframeno, atp_endframeno = (cur_start-ref_start), (cur_end-ref_start)
            merged_startframeno, merged_endframeno = (min(mot_startframeno, atp_startframeno), max(mot_endframeno, atp_endframeno))
            all_merged_bboxes = []
            for relative_frameno in range(merged_startframeno, merged_endframeno+1):
                ### get absolute frame no. using the offset ref_start
                video_frame_no = relative_frameno + ref_start
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
                    all_merged_bboxes.append( np.array([xmin, ymin, xmax, ymax, video_frame_no]))
                else:
                    all_bboxes = np.vstack(trdf['bboxes'][moti])
                    bbox_framenos = all_bboxes[:,-1]
                    mot_boxi = np.where(bbox_framenos==relative_frameno)[0]
                    if(len(mot_boxi)>0):
                        all_merged_bboxes.append( np.concatenate((all_bboxes[mot_boxi[0],:4], [video_frame_no])))
            all_merged_bboxes = np.vstack(all_merged_bboxes)
            ### Note that we are storing the aboslute framenos for each merged bbox. (which is valid for the input video)
            merged_startframeno, merged_endframeno = (all_merged_bboxes[0,4], all_merged_bboxes[-1,4])
            timeframes_length = all_merged_bboxes.shape[0]
            actual_length = merged_endframeno - merged_startframeno + 1
            no_of_missing_frames = actual_length - timeframes_length

            motframes_timegap_max = 0
            motframes_timegap_mean = 0
            motframes_timegap_std = 0
            dts = []
            for fi in range(1, timeframes_length):
                dts.append(all_merged_bboxes[fi, 4] - all_merged_bboxes[fi-1, 4])
            if(len(dts)>0):
                motframes_timegap_max = np.max(dts)
                motframes_timegap_mean = np.round(np.mean(dts),1)
                motframes_timegap_std = np.round(np.std(dts),1)

            ### saving all variables
            trackwise_merged_data['atp_trackid'].append(atp_track_id)
            trackwise_merged_data['mot_id'].append(moti)
            if(atpi==ri):
                trackwise_merged_data['is_atp_main_risk_object'].append(1)
            else:
                trackwise_merged_data['is_atp_main_risk_object'].append(0)
            trackwise_merged_data['merged_start_frameno'].append(merged_startframeno)
            trackwise_merged_data['merged_end_frameno'].append(merged_endframeno)
            trackwise_merged_data['no_of_missing_mergedframes'].append(no_of_missing_frames)
            trackwise_merged_data['mergedframes_timegap_max'].append(motframes_timegap_max)
            trackwise_merged_data['mergedframes_timegap_mean'].append(motframes_timegap_mean)
            trackwise_merged_data['mergedframes_timegap_std'].append(motframes_timegap_std)
            trackwise_merged_data['merged_bboxes'].append(all_merged_bboxes)
            trackwise_merged_data['atp_explanation'].append(atp_explanation)
            trackwise_merged_data['atp_class'].append(atp_obj_class)
            trackwise_merged_data['mot_class'].append(motobj_class)
            trackwise_merged_data['atp_mot_maxiou'].append(motobj_iou)


        showtracks_df = copy.deepcopy(trackwise_merged_data)
        showtracks_df.pop('merged_bboxes')
        showtracks_df = pd.DataFrame(showtracks_df)
        showtracks_df.to_csv(trackdata_csv_path.replace('_nobox_df.csv', '_atpmerged_nobox_df.csv'), index=False)
        np.save(trackdata_path.replace('_dict.npy','_atpmerged_dict.npy'), trackwise_merged_data)
    
