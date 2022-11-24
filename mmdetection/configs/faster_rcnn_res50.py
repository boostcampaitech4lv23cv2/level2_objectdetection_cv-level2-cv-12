_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection wrapper.py',
    '_base_/schedules/schedule_1x.py', 
    './default_runtime.py',
    './logger.py'
]