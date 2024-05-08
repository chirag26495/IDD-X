model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        in_channels=10),
    cls_head=dict(
        type='TSNHead',
        num_classes=3,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.0,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth'
resume_from = None
workflow = [('train', 5)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'RawframeDataset'
data_root = ''
data_root_val = ''
ann_file_train = 'iddx_clips_rawframe_front/train_flow/annotations.txt'
ann_file_val = 'iddx_clips_rawframe_front/val_flow/annotations.txt'
ann_file_test = 'iddx_clips_rawframe_front/val_flow/annotations.txt'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=5, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='RandomResizedCropWidthBounds',
        area_range=(0.5640432098765432, 1.0),
        aspect_ratio_range=(1.5925925925925926, 2.823529411764706),
        min_width_ratio=0.844),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', mean=[128, 128], std=[128, 128], to_bgr=False),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=(420, 208)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', mean=[128, 128], std=[128, 128], to_bgr=False),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Normalize', mean=[128, 128], std=[128, 128], to_bgr=False),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=9,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RawframeDataset',
        ann_file='iddx_clips_rawframe_front/train_flow/annotations.txt',
        data_prefix='',
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        start_index=0,
        pipeline=[
            dict(
                type='SampleFrames', clip_len=5, frame_interval=1,
                num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='RandomResizedCropWidthBounds',
                area_range=(0.5640432098765432, 1.0),
                aspect_ratio_range=(1.5925925925925926, 2.823529411764706),
                min_width_ratio=0.844),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[128, 128],
                std=[128, 128],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW_Flow'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        sample_by_class=False,
        num_classes=3),
    val=dict(
        type='RawframeDataset',
        ann_file='iddx_clips_rawframe_front/val_flow/annotations.txt',
        data_prefix='',
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        start_index=0,
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=5,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=(420, 208)),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128],
                std=[128, 128],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW_Flow'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        num_classes=3),
    test=dict(
        type='RawframeDataset',
        ann_file='iddx_clips_rawframe_front/val_flow/annotations.txt',
        data_prefix='',
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        start_index=0,
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=5,
                frame_interval=1,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=(420, 208)),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128],
                std=[128, 128],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW_Flow'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        num_classes=3))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[60, 120])
total_epochs = 40
work_dir = './work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_iddx_front_IB/'
total_behavior_classes = 3
class_balanced_sampling_strategy = False
init_classifier_from_scratch = False
gpu_ids = [0]
omnisource = False
module_hooks = []
