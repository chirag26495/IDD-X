######### Step7_prepare_rawframedata_annotationsfile.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import random, json
from tqdm import tqdm
from glob import glob

### replace below path to downloaded data folder
relative_src_path = "./"  # "../IDDX/"

### annotation json read
with open(relative_src_path+"iddx_annotations.json", "r") as outfile:
	iddx_annos_json = outfile.read()
iddx_annos_json = json.loads(iddx_annos_json)

egovdrive_behaviors = ['Slowing Down', 'Deviate', 'Turning and Slowing Down']

iddx_clips_dir = 'iddx_clips_rawframe_'
pwd_dir = os.getcwd()+'/'

all_eventids = []
all_egoactions = []
for eventi in iddx_annos_json:
    all_eventids.append(eventi['event_id'])
    all_egoactions.append(eventi['ego-vehicle_driving_behavior'])

####################################################################################################################################################
for dstype in ['train', 'val', 'test']:
    for viewi in ['front/', 'rear/']:
        for vmodal in ['', '_flow']:   
            clips_list = []
            cliplength_list = []
            cliplabel_list = []
            iddx_clips_ds_dir = iddx_clips_dir + viewi + dstype + vmodal + '/Scenes/'
            if(not os.path.exists(iddx_clips_ds_dir)):
                continue
            out_clip_dirs = glob(iddx_clips_ds_dir+'*')
            for out_clip_dir in out_clip_dirs:
                if(not os.path.exists(out_clip_dir)):
                    continue
                ego_eventid = int(out_clip_dir.split('/')[-1].split('_')[0])
                ego_action_idx = egovdrive_behaviors.index(all_egoactions[all_eventids.index(ego_eventid)])
                
                clip_frame_count = len(os.listdir(out_clip_dir))
                if(vmodal == '_flow'):
                    clip_frame_count = clip_frame_count//2
                if(clip_frame_count > 0):
                    cliplength_list.append(clip_frame_count)
                    clips_list.append(pwd_dir+out_clip_dir)
                    cliplabel_list.append(ego_action_idx)

            annot_df = pd.DataFrame({0:clips_list, 1:cliplength_list, 2:cliplabel_list})
            annot_df.to_csv(iddx_clips_ds_dir.replace('/Scenes/', '')+'/annotations.txt', header=False, index=False, sep=' ')
        
# annot_df