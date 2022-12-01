import os
import glob

from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

#from utils import csv_preprocess
from pycocotools.coco import COCO


class CustomDataset(Dataset):  # for train and validation
    def __init__(self, gt_path, data_path='/opt/ml/dataset', transform=None, phase="train"):
        self.coco = COCO(gt_path) #'/opt/ml/dataset/kfold/seed41/train_4.json'
        self.data_path = data_path
        self.transform = transform
        self.img_ids = self.coco.getImgIds()
        self.phase= phase
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids = self.img_ids[idx]
        image_info = self.coco.loadImgs(img_ids)[0]['file_name']
        image = Image.open(os.path.join(self.data_path,image_info))
        if self.transform:
            image = self.transform(image)

        if self.phase == "test":
            return image,image_info

        annIds = self.coco.getAnnIds(imgIds=img_ids,iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        label = list(set([i['category_id'] for i in anns]))
        # if not label:
        #     label = [10]
        label = torch.tensor(label)
        #label = torch.sum(torch.nn.functional.one_hot(label,num_classes=11),dim=0)
        label = torch.sum(torch.nn.functional.one_hot(label,num_classes=10),dim=0)
        return image, label

if __name__ == "__main__":
    # train_dir = "/opt/ml/input/data/train"
    # val_ratio = 0.3
    batch_size = 16
    seed = 42

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=512, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=(10,10)),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # coco = COCO('/opt/ml/dataset/kfold/seed41/train_4.json')
    # ind = 1001
    # img_ids = coco.getImgIds()[ind]
    # image_info = coco.loadImgs(img_ids)[0]['file_name']
    # annIds = coco.getAnnIds(imgIds=img_ids,iscrowd=None)
    # anns = coco.loadAnns(annIds)

    # print(img_ids,image_info)
    # print(anns)
    # print([i['category_id'] for i in anns])


    # train_data, val_data = train_test_split(
    #     data, test_size=val_ratio, shuffle=True, random_state=seed
    # )
    train_gt = '/opt/ml/dataset/kfold/seed41/train_4.json'
    train_dataset = CustomDataset(gt_path=train_gt, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for img, label in train_loader:
        print(img.shape, label.shape)
        break

    # tmp = iter(train_dataset)
    # img, label = next(tmp)