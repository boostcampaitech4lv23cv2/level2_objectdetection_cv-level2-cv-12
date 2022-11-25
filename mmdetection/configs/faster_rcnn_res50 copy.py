_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', 
    './default_runtime.py',
    './logger.py'
]

# model = dict(
#     test_cfg=dict(
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='soft_nms', iou_threshold=0.5),
#             max_per_img=100)))