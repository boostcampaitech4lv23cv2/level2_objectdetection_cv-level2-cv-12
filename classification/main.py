import os
import argparse
import random
from datetime import datetime
from collections import defaultdict

import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import BCELoss
import torch.nn.functional as F

from dataset import CustomDataset
from model import CustomModel
from utils import set_seed, build_transform
import glob
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=2)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--backbone_name", type=str, default="efficientnet_b3")

    parser.add_argument("--save_dir", type=str, default="/opt/ml/level2_objectdetection_cv-level2-cv-12/classification/work_dirs")
    parser.add_argument("--project_name", type=str, default="classification")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="efficientnet_b3",
    )
    parser.add_argument("--image_scale", type=tuple, default=(1024, 1024))
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    save_path = os.path.join(
        args.save_dir, args.project_name, args.experiment_name
    )
    
    if os.path.exists(save_path):
        save_path += '_' + str(len(glob.glob(save_path+'*')) + 1)
    
    wandb.init(project=args.project_name, name=args.experiment_name, entity="cv12")
    wandb.config.update(args)
    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/config.json', 'w') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=4)

    train_transform, val_transform = build_transform(args=args, phase="train", image_scale=args.image_scale)
    train_dataset = CustomDataset(gt_path='/opt/ml/dataset/kfold/seed41/train_4.json', transform=train_transform)
    val_dataset = CustomDataset(gt_path='/opt/ml/dataset/kfold/seed41/val_4.json', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.n_workers)

    model = CustomModel(args).cuda()
    loss_fn = BCELoss().cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, eta_min=args.lr*0.1, T_max=10)

    best_epoch, best_score = 0, 0
    for epoch in range(1, args.num_epochs + 1):
        print("\n### epoch {} ###".format(epoch))
        info, time = defaultdict(int), datetime.now()
        
        losses = []
        model.train()
        # for img, label in train_loader:
        for img, label in tqdm(train_loader):
            img, label = img.cuda(), label.float().cuda()
            
            optimizer.zero_grad()
            pred = model(img).sigmoid()
            loss = loss_fn(pred,label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        info["train_total_loss"] = np.mean(losses)
    
        elapsed = datetime.now() - time
        info["epoch"] = epoch
        info["learning_rate"] = optimizer.param_groups[0]['lr']
        scheduler.step()
        print("[train] loss {:.3f} | elapsed {}".format(info["train_total_loss"], elapsed))
        wandb.log(info)

        ### validation ###
        preds, labels = torch.tensor([]), torch.tensor([])
        info, time = defaultdict(int), datetime.now()
        val_losses = []
        model.eval()
        with torch.no_grad():
            #for img, label in val_loader:
            for img, label in tqdm(val_loader):
                img, label = img.cuda(), label.float().cuda()
                pred = model(img).sigmoid()
                loss = loss_fn(pred,label)
                preds = torch.cat((preds, pred.cpu()))
                labels = torch.cat((labels, label.cpu()))
                val_losses.append(loss.item())
            info["val_total_loss"] = np.mean(val_losses)
            
            info["epoch"] = epoch
            all_auc = roc_auc_score(labels.tolist(), preds.tolist(),average=None)
            info["auc"] =sum(all_auc)/len(all_auc)
            elapsed = datetime.now() - time

            print("[val] loss {:.3f} | auc {:.3f} | elapsed {}".format(info["val_total_loss"], info["auc"], elapsed))
            print("[val] {}".format([float(str(i)[:5]) for i in all_auc]))
            wandb.log(info)

        ### save model ###
        if best_score < info["auc"]:
            best_epoch, best_score = epoch, info["auc"]
            torch.save(
                {"model": model.state_dict()},
                os.path.join(save_path, "best_model.pth"),
            )
            print(">>>>>> SAVED model at {:02d}".format(epoch))
        print("[best] epoch: {}, score : {:.4f}".format(best_epoch, best_score))
    wandb.finish()