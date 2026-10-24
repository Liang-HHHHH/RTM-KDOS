_base_ = ['mmpose::_base_/default_runtime.py']

# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = "D:/Python/mm/mmpose/data2/plant_coco/"

# 三角板关键点检测数据集-元数据
dataset_info = {
    'dataset_name':'plant_coco',
    'classes':('leaf', 'fruit'),
    'keypoint_info':{
        0:{'name':'kp1','id':0,'color':[255,0,0],'type': '','swap': ''},
        1:{'name':'kp2','id':1,'color':[0,255,0],'type': '','swap': ''},
        2:{'name':'kp3','id':2,'color':[0,0,255],'type': '','swap': ''}
    },
    'skeleton_info': {
        0: {'link':('kp1','kp2'),'id': 0,'color': [100,150,200]},
        1: {'link':('kp2','kp3'),'id': 1,'color': [200,100,150]},
    },
    'joint_weights':[1.0, 1.0, 1.0],
    'sigmas':[0.026,0.025,0.025]
}

NUM_KEYPOINTS = len(dataset_info['keypoint_info'])

# 训练超参数
max_epochs = 400
val_interval = 10
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 64
val_batch_size = 32
stage2_num_epochs = 10
base_lr = 0.0004
randomness = dict(seed=42)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
)

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=20),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=100,
        end=max_epochs,
        T_max=max_epochs - 100,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5, 5),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)


## 模型：RTMPose-S with Improved Scoring Head
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        use_simam=False,
        use_dsconv=False,
        dsconv_stages=[],
        dsconv_kwargs=dict(
            kernel_size=3,
            extend_scope=1,
            morph=1,
            if_offset=True,
            device='cuda'
        ),
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/' +
            'rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth',
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        in_featuremap_size=(12, 12),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        # Keypoint loss
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
        ),
        score_loss=dict(
            type='SmoothL1Loss',
            loss_weight=0.5
        ),
        use_geometric_features=True,
        decoder=codec),
    test_cfg=dict(flip_test=False))


backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

test_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]


# Stage 2 Pipeline - 移除RandomHalfBody保持三点完整
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
    ))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='test_coco.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline,
    ))

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),
]

test_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'test_coco.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),
]