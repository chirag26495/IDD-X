###
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import random, json
from tqdm import tqdm

### replace below path to downloaded data folder
relative_src_path = "./"  # "../IDDX/"

### annotation json read
with open(relative_src_path+"iddx_annotations.json", "r") as outfile:
	iddx_annos_json = outfile.read()
iddx_annos_json = json.loads(iddx_annos_json)

egovdrive_behaviors = ['Slowing Down', 'Deviate', 'Turning and Slowing Down']
# explanation_labels = ['Congestion', 'Confrontation', 'Slowing Down', 'Overtake', 'Avoid Congestion', 'Obstruction', 'Crossing', 'On-road Being', 'Cut-in', 'Avoid On-road Being', 'Avoid Obstruction', 'Stopped Vehicle', 'Deviate', 'Avoid Stopped Vehicle', 'Merging', 'Red Light', 'Left Turn', 'U-Turn', 'Right Turn']

video_src_dir = relative_src_path+'iddx_all_videos/'
iddx_clips_front_dir = 'iddx_clips_rawframe_front/'
iddx_clips_rear_dir = 'iddx_clips_rawframe_rear/'
pwd_dir = os.getcwd()+'/'

####################################################################################################################################################
clips_list = []
clips_rear_list = []
cliplength_list = []
cliplabel_list = []

for eventi in iddx_annos_json:
# if(True):
#     eventi = iddx_annos_json[0]
    dstype = eventi['data']
    iddx_clips_ds_dir = iddx_clips_front_dir+dstype+'/Scenes/'
    os.makedirs(iddx_clips_ds_dir, exist_ok=True)
    iddx_clips_ds_rear_dir = iddx_clips_rear_dir+dstype+'/Scenes/'
    os.makedirs(iddx_clips_ds_rear_dir, exist_ok=True)

    ref_start, ref_end = eventi['start_frame'], eventi['end_frame']
    ego_action = eventi['ego-vehicle_driving_behavior']
    ego_video_name = eventi['video_name']
    ego_event_id = eventi['event_id']

    out_clip_dir = iddx_clips_ds_dir+str(ego_event_id)+'_'+ego_video_name.replace('_0060_combined.mp4', '')+'_'+str(ref_start)+'to'+str(ref_end)+'/'
    # out_clip_dir = iddx_clips_ds_dir+str(ego_task_id)+'_'+str(ego_event_id)+'_'+str(ref_start)+'to'+str(ref_end)+'/'
    os.makedirs(out_clip_dir, exist_ok=True)
    out_clip_rear_dir = iddx_clips_ds_rear_dir+str(ego_event_id)+'_'+ego_video_name.replace('_0060_combined.mp4', '')+'_'+str(ref_start)+'to'+str(ref_end)+'/'
    os.makedirs(out_clip_rear_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_src_dir+ego_video_name)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file", video_src_dir+ego_video_name)
    video_frame_no = -1
    clip_frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            video_frame_no+=1
            if(video_frame_no>=ref_start and video_frame_no<=ref_end):
                clip_frame_count+=1
                front_img = frame[0:2160//2, 0:1920]
                rear_img = frame[2160//2:, 0:1920]
                if(not os.path.exists(out_clip_dir+'img_{0:05d}.jpg'.format(clip_frame_count))):
                    cv2.imwrite(out_clip_dir+'img_{0:05d}.jpg'.format(clip_frame_count), front_img)
                if(not os.path.exists(out_clip_rear_dir+'img_{0:05d}.jpg'.format(clip_frame_count))):
                    cv2.imwrite(out_clip_rear_dir+'img_{0:05d}.jpg'.format(clip_frame_count), rear_img)
        else:
            break
    cap.release()
    clips_list.append(pwd_dir+out_clip_dir[:-1])
    clips_rear_list.append(pwd_dir+out_clip_rear_dir[:-1])
    cliplength_list.append(clip_frame_count)
    cliplabel_list.append(egovdrive_behaviors.index(ego_action))

# annot_df = pd.DataFrame({0:clips_list, 1:cliplength_list, 2:cliplabel_list})
# annot_df.to_csv(iddx_clips_front_dir+dstype+'/annotations.txt', header=False, index=False, sep=' ')
# annot_df_rear = pd.DataFrame({0:clips_rear_list, 1:cliplength_list, 2:cliplabel_list})
# annot_df_rear.to_csv(iddx_clips_rear_dir+dstype+'/annotations.txt', header=False, index=False, sep=' ')

### Overall stats
action_labels_stats = {'': ['#scenarios','#median_clip_len']}     ### '#avg_clip_len', 
for ari, arl in enumerate(egovdrive_behaviors):
    tmp = annot_df.loc[annot_df[2]==ari, 1]
    action_labels_stats[arl] = [str(len(tmp)), round(tmp.median())]   ### round(tmp.mean()), 

behav_stats = pd.DataFrame(action_labels_stats)
behav_stats.to_csv("Behavior_summary.csv",index=False)
print(behav_stats)

