### Step3_MOTdata_extract.py
import sys
root_dir = 'IDD-D_Yolo-v4-Model/'
sys.path.insert(0,root_dir+'pytorch-YOLOv4/')
sys.path.insert(0,root_dir)

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse
import numpy as np, cv2, pandas as pd
from glob import glob as gb
import gc, copy
from sort.sort import *
from tqdm import tqdm

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_detmatch(query_trackbox, detboxes, matched_det_idxs):
    matched_det_idx = -1
    matched_det_iou = 0
    for detidx in range(detboxes.shape[0]):
        ### if already matched with some other
        if(detidx in matched_det_idxs):
            continue
        qiou = bb_intersection_over_union(query_trackbox, detboxes[detidx, :4])
        ### min. matching thresh for query_trackbox with detected bbox = 0.7 (can be tuned)
        if(qiou > 0.7 and matched_det_iou < qiou):
            matched_det_idx = detidx
            matched_det_iou = qiou
    return(matched_det_idx, matched_det_iou)

device = torch.device('cuda:0')
cfgfile = root_dir + 'idd.cfg' 
weightfile = root_dir + 'idd_best.weights' 
conf_thresh = 0.4
nms_thresh = 0.6
batch_size = 5

model = Darknet(cfgfile)
# model.print_network()
model.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))
model.to(device)
# model = torch.nn.DataParallel(model)
model.eval()

num_classes = model.num_classes
namesfile = root_dir + 'idd.names'
class_names = load_class_names(namesfile)

dstypes = ['train', 'val']
for dstype in dstypes:
    # dstype = 'train'
    
    outdir = 'iddd_trackwise_'+dstype+'_data/'
    os.makedirs(outdir, exist_ok=True)

    img_wd, img_ht = (1920, 1080)
    dirls = gb('iddx_clips_rawframe_front/'+dstype+'/Scenes/*')
    # for scenedir in tqdm(dirls[223:]):
    # for scenedir in tqdm(dirls[54+73+13:]):
    dirls.reverse()
    for scenedir in tqdm(dirls):

    # ### debug mode
    # if(True):
    #     debug_index = 0
    #     scenedir = dirls[debug_index]

        scenario_dir_nm = scenedir.split('/')[-1]
        if(os.path.exists(outdir + scenario_dir_nm +'_dict.npy')):
            # print()
            continue
        scenedir = scenedir+'/'
        # scenedir = '/ssd_scratch/cp_wks/codes/IDDX/iddx_clips_rawframe_front/val/Scenes/2396_123_1364to1394/'
        imgfiles = gb(scenedir+'*.jpg')
        if(len(imgfiles)==0):
            continue
        imgfiles.sort()
        resized_imgs = None
        for imgfile in imgfiles:
            img = cv2.imread(imgfile)
            # print(model.width, model.height)
            resized_img = cv2.resize(img, (model.width, model.height))
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            resized_img = torch.from_numpy(resized_img.transpose(2, 0, 1)).float().div(255.0)
            if(resized_imgs is None):
                resized_imgs = resized_img.unsqueeze(0)
            else:
                resized_imgs = torch.vstack([resized_imgs, resized_img.unsqueeze(0)])
        # print(resized_imgs.shape)
        resized_imgs = resized_imgs.to(device)
        resized_imgs = torch.autograd.Variable(resized_imgs)

        ############ DETECTION
        all_boxes = []
        for idxs in range(0, resized_imgs.shape[0], batch_size):
            with torch.no_grad():
                output = model(resized_imgs[idxs : idxs+batch_size, ...])
                boxes = utils.post_processing(resized_imgs[idxs : idxs+batch_size, ...], conf_thresh, nms_thresh, output)
                ### deleting exxtra conf val, and omitting if empty box found
                boxes_ = []
                for bi in boxes:
                    if(len(bi)==0):
                        continue
                    boxes_.append(np.delete(np.array(bi), 4, axis=1))
                boxes = boxes_
                all_boxes.extend(boxes)
        # print(len(all_boxes), all_boxes[1].shape)

        ############ TRACKING
        ### delete _ variables, clear memory and reinit trackid val before running SORT on fresh video 
        # prev_vars = set(dir())
        # for obj in new_vars - prev_vars:
        #     if (not obj.startswith('__') and obj.startswith('_')):# and obj!='all_boxes_with_trackids' and obj!='prev_vars' and obj!='new_vars'):
        #         del globals()[obj]
        # gc.collect()
        KalmanBoxTracker.count = 0

        ### create instance of SORT  : Fix bug of re-initializing the tracker for a new scene
        mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3) 

        all_boxes_with_trackids = []
        for detections in all_boxes:
            ### remove the obj class, this is not required for SORT algo; ---> assigning track with correct class is Fixed later via iou matching
            detections = np.delete(detections, -1, axis=1)
            # update SORT
            track_bbs_ids = mot_tracker.update(detections)
            all_boxes_with_trackids.append(track_bbs_ids)
        # print(len(all_boxes_with_trackids))

        ### Prepare trackwise data
        tracks_ids = {}
        for frameno, perframe_detected_tracks in enumerate(all_boxes_with_trackids):
            matched_det_idxs_ = []
            for dti in range(perframe_detected_tracks.shape[0]):
                query_trackbox_ = perframe_detected_tracks[dti, :]
                detboxes_ = all_boxes[frameno]
                matched_det_idx_, _ = get_detmatch(query_trackbox_, detboxes_, matched_det_idxs_)
                matched_det_idxs_.append(matched_det_idx_)
                dti_conf, dti_objclassid = (0, -1)
                if(matched_det_idx_ != -1):
                    # print(matched_det_idx_, _, all_boxes[frameno][matched_det_idx_, :], query_trackbox_)
                    dti_conf, dti_objclassid = all_boxes[frameno][matched_det_idx_, [4, 5]]
                else:
                    ### use any prev. matched detbox data (compy same objclass)
                    print("match not found, copy same obj class if available from prev. frame's match.")

                dti_trackid = int(perframe_detected_tracks[dti, 4])
                if(dti_trackid not in tracks_ids):
                    track_bbox_atframeno = np.hstack([perframe_detected_tracks[dti, :4], dti_conf, dti_objclassid, frameno])
                    tracks_ids[dti_trackid] = np.array([track_bbox_atframeno])
                else:
                    if(dti_objclassid == -1):
                        dti_objclassid = tracks_ids[dti_trackid][-1, 5]
                    track_bbox_atframeno = np.hstack([perframe_detected_tracks[dti, :4], dti_conf, dti_objclassid, frameno])
                    tracks_ids[dti_trackid] = np.vstack([tracks_ids[dti_trackid], track_bbox_atframeno])

        ### tracks_ids[id]: [4 bbox coords , det_conf, det_objclass, scene_frameno]
        trackwise_data = {'trackid':np.array([]), 'start_frameno':[], 'end_frameno':[], 'bboxes':[], 'label':[], 'detboxes_conf':[]}
        for tri in list(tracks_ids.keys()):
            trackwise_data['trackid'] = np.concatenate((trackwise_data['trackid'], [tri]))
            trackwise_data['start_frameno'].append(int(tracks_ids[tri][0, -1]))
            trackwise_data['end_frameno'].append(int(tracks_ids[tri][-1, -1]))
            trackwise_data['detboxes_conf'].append(tracks_ids[tri][:, -3])

            ### scale the bbox coords
            trackboundinboxes = tracks_ids[tri][:, [0,1,2,3,-1]]
            trackboundinboxes[:, [0,2]] = np.clip(trackboundinboxes[:, [0,2]]*img_wd, 0, img_wd)
            trackboundinboxes[:, [1,3]] = np.clip(trackboundinboxes[:, [1,3]]*img_ht, 0, img_ht)
            trackboundinboxes = trackboundinboxes.astype(np.int32)
            trackwise_data['bboxes'].append(trackboundinboxes)

            ### most common label based obj class label assignment to track
            alllabelss = tracks_ids[tri][:, -2]
            labelmedian = np.median(alllabelss)
            if(labelmedian not in alllabelss):
                trackobjlabel = int(alllabelss[0])
            else:
                trackobjlabel = int(labelmedian)
            trackwise_data['label'].append(trackobjlabel)

        showtracks_df = copy.deepcopy(trackwise_data)
        showtracks_df.pop('bboxes')
        showtracks_df.pop('detboxes_conf')
        showtracks_df = pd.DataFrame(showtracks_df)
        showtracks_df.to_csv(outdir + scenario_dir_nm + '_nobox_df.csv',index=False)
        np.save(outdir + scenario_dir_nm +'_dict.npy', trackwise_data)
        
    #     break
    # break
