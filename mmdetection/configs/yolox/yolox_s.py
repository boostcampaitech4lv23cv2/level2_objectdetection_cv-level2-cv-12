#_base_ = ['../_base_/schedules/schedule_1x.py', '../default_runtime.py']
_base_ = ['../default_runtime.py','../logger.py']

img_scale = (640, 640)  # height, width

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=10, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
    #test_cfg=dict(score_thr=0.01, nms=dict(type='soft_nms', iou_threshold=0.65))
    )

# dataset settings
dataset_type = 'CocoDataset'
image_data_root = '/opt/ml/dataset'
annotation_data_root = '/opt/ml/dataset/kfold/seed41/'

# https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
load_from = '/opt/ml/level2_objectdetection_cv-level2-cv-12/work_dirs/pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 1.5), # Min and max ratio of scaling transform
        border=(-img_scale[0] // 2, -img_scale[1] // 2), # Distance from height and width sides of input image to adjust output shape, Only used in mosaic dataset
        ), 
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=annotation_data_root + 'train_4.json',
        img_prefix=image_data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
        classes=classes
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=([128,128],[384,384],[640,640],[960,960]),
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4,
#     persistent_workers=True,
#     train=train_dataset,
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=annotation_data_root + 'val_4.json',
        img_prefix=image_data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=image_data_root + 'test.json',
        img_prefix=image_data_root,
        classes=classes,
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    #type='Adam',
    type='SGD',
    lr=0.001,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 50
num_last_epochs = 15 # The number of latter epochs in the end of the training to close the data augmentation and switch to L1 loss.
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    #_delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=-1) # don't save at every epoch
evaluation = dict(
    save_best='bbox_mAP_50',
    greater_keys=['bbox_mAP_50'],
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    classwise=True,
    metric='bbox')

# log_config = dict(interval=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)