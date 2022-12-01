import random
import numpy as np
import torch
from torchvision import transforms

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transform(args=None, phase="train"):
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    
    if phase == "train":
        train_transform = transforms.Compose(
            [
                #transforms.RandomResizedCrop(size=512, scale=(0.2, 1.0)),
                transforms.Resize(size=(512,512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
         ]
        )
        val_transform = transforms.Compose(
            [
                #transforms.RandomResizedCrop(size=512, scale=(0.2, 1.0)),
                transforms.Resize(size=(512,512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return train_transform, val_transform
    else:
        test_transform = transforms.Compose(
            [
                #transforms.RandomResizedCrop(size=512, scale=(0.2, 1.0)),
                transforms.Resize(size=(512,512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return test_transform