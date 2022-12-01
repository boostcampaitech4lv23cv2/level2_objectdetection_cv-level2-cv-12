_base_ = [
    'cascade_focal/model.py',
    'cascade_focal/datasets.py',
    'cascade_focal/schedule.py', 
    'default_runtime.py',
    'logger.py'
]

load_from = '/opt/ml/baseline/mmdetection/pretrained/focalnet_tiny_lrf_cascade_maskrcnn_3x.pth'

model = dict(
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[11,9,9,7],        
        focal_levels=[3,3,3,3], 
        use_conv_embed=False, 
    ))
