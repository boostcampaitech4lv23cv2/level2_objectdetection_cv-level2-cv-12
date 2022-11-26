# sangho 20221124
# https://stages.ai/competitions/218/discussion/talk/post/1817
# example. python confusion_matrix.py --gt_json {dir}/gt.json --pred_csv {dir}/prediction.csv --file_name confusion_matrix.png --norm True
# options
# 1. normalize: display 값 설정. --norm True: ratio, False: number
#               ratio는 GT 기준, class별 개수를 나눈 값 (그림에서 column마다 같은 값을 나눈다.)
# 2. CONF_THRESHOLD: confusion matrix에 추가할 sample의 class score threshold (default=0.01). line 153에서 조정
# 3. IOU_THRESHOLD: confusion matrix에 추가할 sample의 IoU threshold (default=0.5). line 153에서 조정
#                   TH보다 큰 IoU의 box에 대해서만 matrix에 count된다.

import numpy as np
import argparse
from pycocotools.coco import COCO
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def box_iou_calc(boxes1, boxes2):
    # <https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py>
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def plot(self, file_name='./', norm=True, names=["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]):
        try:
            import seaborn as sn

            if norm: 
                array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize: GT 기준!
                fmtt = '.2f'
                array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            else: 
                array = self.matrix.astype(int)
                fmtt = 'g'
            
            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt=fmtt, square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            
            fig.axes[0].text(1.07, 1.03, 'Matrix\nConfidence TH=%.2f \nIoU TH=%.2f' %(self.CONF_THRESHOLD, self.IOU_THRESHOLD), 
                                verticalalignment='bottom', horizontalalignment='left', 
                                transform=fig.axes[0].transAxes, fontsize=10)
            fig.axes[0].set_xlabel('GT')
            fig.axes[0].set_ylabel('Predicted')
            fig.axes[0].xaxis.tick_top()
            fig.axes[0].xaxis.set_tick_params(top=False)
            for tick in fig.axes[0].get_xticklabels():
                tick.set_rotation(30)
            fig.savefig(file_name, dpi=250)
        except Exception as e:
            print(e)
            pass

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])     # output dim.: (N_gt, N_prediction)
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)            # want_idx[0]: idx_gt, want_idx[1]: idx_gt에 대응되는 idx_pred

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        # 일종의 NMS? gt box, pred box에 대해서 각각 중복 매칭되는 것을 지우는 듯
        # np.unique(arr, return_index=True) -> arr에서 처음 나오는 element의 indices도 return
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]
        # all_matches = [[idx_gt, idx_pred, IoU], ~]?

        # gt마다 대응되는 prediction이 있으면 conf_matrix에 추가, 없으면(else) matrix 마지막에 추가
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        # 위에서 빠진 case에 대한 코드
        # prediction(detection)마다,
        # gt에 대응되는 prediction이 없거나(=gt-pred pair가 없다), 있더라도 해당 prediction에 대한 것은 없다면, matrix 마지막에 추가
        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

def main(args):
    conf_mat = ConfusionMatrix(num_classes = 10, CONF_THRESHOLD = 0.01, IOU_THRESHOLD = 0.5)
    # CONF_THRESHOLD: threshold for class confidence (prediction)
    # IOU_THRESHOLD: threshold for IoU. TH보다 큰 IoU의 box에 대해서만 matrix에 count된다.

    # load gt
    gt_path = args.gt_json
    pred_path = args.pred_csv
    with open(gt_path, 'r') as outfile:
        test_anno = (json.load(outfile))

    # load prediction
    pred_df = pd.read_csv(pred_path)

    new_pred = []
    gt = []

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()

    # empty box check
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')
    
    # collect predicted boxes
    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        new_pred.append([])
        
        # box split
        boxes = np.array(str(bbox).split(' '))
        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)       # 마지막에 빈 값('')이 하나 생긴다.
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        
        for box in boxes:
            new_pred[-1].append([float(box[2]), float(box[3]), float(box[4]), float(box[5]),  float(box[1]), float(box[0])])

    # collect gts
    coco = COCO(gt_path)
    for image_id in coco.getImgIds():
        gt.append([])
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)
        file_name = image_info['file_name']

        for ann in anns:
            gt[-1].append([
                       float(ann['category_id']), 
                       float(ann['bbox'][0]), 
                       float(ann['bbox'][1]),
                       float(ann['bbox'][0]) + float(ann['bbox'][2]),
                       (float(ann['bbox'][1]) + float(ann['bbox'][3])),])

    # example
    # new_pred[2]: [[705.9431, 196.75587, 983.2462, 417.29224, 0.27008054, 0.0], 
    #               [7.530319, 0.21073914, 306.00677, 72.708786, 0.13421041, 0.0], 
    #               [422.08997, 395.3664, 855.5221, 703.9649, 0.8417508, 2.0], 
    #               [419.01917, 381.32715, 854.8352, 713.45776, 0.07396625, 3.0], 
    #               [710.2981, 197.71304, 988.6636, 413.22708, 0.10487595, 7.0]]
    # gt[2]: [[6.0, 844.2, 499.5, 892.5, 600.7], [3.0, 423.9, 390.4, 848.0, 717.8], [6.0, 885.6, 577.0, 957.0, 659.6]]
    i=0
    for p, g in zip(new_pred, gt):
        conf_mat.process_batch(np.array(p), np.array(g))

    conf_mat.plot(args.file_name, norm=args.norm=='True')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gt_json', type=str)
    parser.add_argument('--pred_csv', type=str)
    parser.add_argument('--file_name', type=str, default='confusion_matrix.png',)
    parser.add_argument('--norm', type=str, default=True)
    args = parser.parse_args()
    main(args)