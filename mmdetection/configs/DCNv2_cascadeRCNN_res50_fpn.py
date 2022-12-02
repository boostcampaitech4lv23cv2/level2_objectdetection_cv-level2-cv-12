# https://github.com/open-mmlab/mmdetection/tree/master/configs/dcnv2
_base_ = [
    '_base_/models/cascade_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',      #'_base_/datasets/coco_detection_resize.py' # '_base_/datasets/coco_detection_multiscale.py',      #,
    '_base_/schedules/schedule_1x.py', 
    './default_runtime.py',
    './logger.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth'