import os
import pandas as pd
import torch
import torch.nn as nn

import timm


class CustomModel(nn.Module):
    def __init__(self, args=None):
        super(CustomModel, self).__init__()

        num_classes = 10
        self.backbone = timm.create_model(
            args.backbone_name, pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
