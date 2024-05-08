### Step2_get_flowframes.py
import os
import os.path as osp
import multiprocessing

method = 'tvl1'
def opflow(full_path):
    method = 'tvl1'
    out_full_path = 'iddx_clips_rawframe_front/' + full_path.split('/')[-3] + '_flow/Scenes'
    out_full_path_wdir = out_full_path+'/'+full_path.split('/')[-1]
    
    iretry = 0
    tmp = 0
    while(True):
        if(iretry>=4):
            break
        if(os.path.exists(out_full_path_wdir)):
            if(len(os.listdir(out_full_path_wdir))//2 == len(os.listdir(full_path))-1):
                break
        #### see the out path it should have the scene folder path also
        cmd = osp.join(f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'" f' -v --if')
        # print(cmd)
        tmp = os.system(cmd)
        # # print(tmp)
        iretry+=1
        if(iretry>3):
            print(f"### Retrying-{iretry} for: ",full_path)
    return tmp


incomplete_events = []
num_trys = 4
view_dir = 'iddx_clips_rawframe_front/'
alldirs_forflow = []
for dsplit in ['train', 'val']:
    rootdir = view_dir + dsplit + '/Scenes/'
    # out_full_path = view_dir + dsplit + '_flow/Scenes'
    # os.makedirs(out_full_path, exist_ok=True)
    
    alldirs = os.listdir(rootdir)
    for di in alldirs:
        full_path = rootdir+di
        
        out_full_path = 'iddx_clips_rawframe_front/' + full_path.split('/')[-3] + '_flow/Scenes'
        out_full_path_wdir = out_full_path+'/'+full_path.split('/')[-1]
        if(os.path.exists(out_full_path_wdir)):
            if(len(os.listdir(out_full_path_wdir))//2 != len(os.listdir(full_path))-1 and len(os.listdir(full_path))>1):
                incomplete_events.append(out_full_path_wdir)
                alldirs_forflow.append(full_path)
        

### 19 process for train, 4 for val, total 23
pool_obj = multiprocessing.Pool(14)
ans = pool_obj.map(opflow, alldirs_forflow)
print(ans)
pool_obj.close()
