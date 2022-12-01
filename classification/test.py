import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import CustomModel
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--backbone_name", type=str, default="efficientnet_b0")
    parser.add_argument("--checkpoint_path", type=str, default="/opt/ml/level2_objectdetection_cv-level2-cv-12/classification/work_dirs/classification/efficientnet_b2/best_model.pth")
    parser.add_argument("--result_csv", type=str, default="/opt/ml/level2_objectdetection_cv-level2-cv-12/classification/work_dirs/classification/efficientnet_b2/test_class.csv")

    parser.add_argument("--image_scale", type=tuple, default=(1024, 1024))
    args = parser.parse_args()
    set_seed(args.seed)
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    test_transform = build_transform(args=args, phase="test", image_scale=args.image_scale)
    test_dataset = CustomDataset(gt_path='/opt/ml/dataset/test.json', transform=test_transform, phase="test")
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.n_workers)

    model = CustomModel(args).to(device)
    checkpoint = torch.load(args.checkpoint_path,map_location=device)['model']
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint)

    model.eval()
    file_names = []
    prediction_strings = []
    preds = torch.tensor([])
    
    with torch.no_grad():
        for img,image_info in tqdm(test_loader):
            img,image_info = img.to(device), image_info
            file_names.extend(image_info)
            pred = model(img).sigmoid()
            preds = torch.cat((preds, pred.cpu()))
        
    preds = preds.numpy()
    preds = [" ".join(map(str,i)) for i in preds]
    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['classe_str'] = preds
    submission.to_csv(args.result_csv, index=None)