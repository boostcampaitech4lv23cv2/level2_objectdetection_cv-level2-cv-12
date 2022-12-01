import os
import argparse
import random
from datetime import datetime
from collections import defaultdict

import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

from dataset import CustomDataset
from model import CustomModel
from utils import *

# from timm.scheduler.step_lr import StepLRScheduler
#from metric import accuracy, macro_f1

#from process import train, validation
#from loss import MultitaskLoss, create_criterion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--backbone_name", type=str, default="efficientnet_b0")
    parser.add_argument("--checkpoint_path", type=str, default="work_dirs/classification/efficientnet_b0/best_model.pth")
    parser.add_argument("--result_csv", type=str, default="work_dirs/classification/efficientnet_b0/test_class.csv")


    args = parser.parse_args()
    set_seed(args.seed)

    test_transform = build_transform(args=args, phase="test")
    #test_dataset = CustomDataset(gt_path='/opt/ml/dataset/kfold/seed41/val_4.json', transform=test_transform)
    test_dataset = CustomDataset(gt_path='/opt/ml/dataset/test.json', transform=test_transform, phase="test")
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.n_workers)

    model = CustomModel(args).cuda()
    checkpoint = torch.load(args.checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)

    model.eval()
    submission = pd.DataFrame()
    # submission['PredictionString'] = prediction_strings
    # submission['image_id'] = file_names
    preds = torch.tensor([])
    with torch.no_grad():
        for img in tqdm(test_loader):
            img,image_info = img.cuda(),image_info
            pred = model(img).sigmoid()
            preds = torch.cat((preds, pred.cpu()))
    
    submission.to_csv(args.result_csv, index=None)