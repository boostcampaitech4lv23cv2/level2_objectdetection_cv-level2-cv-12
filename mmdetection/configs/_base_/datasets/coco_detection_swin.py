# dataset settings
dataset_type = 'CocoDataset'

image_data_root = '/opt/ml/dataset'
annotation_data_root = '/opt/ml/dataset/kfold/seed41/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=annotation_data_root + 'train_4.json',
        img_prefix=image_data_root,
        classes=classes,
        pipeline=train_pipeline
        ),
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

evaluation = dict(
    save_best='bbox_mAP_50',
    greater_keys=['bbox_mAP_50'],
    interval=1,
    classwise=True,
    metric='bbox'
    )