import os
import pandas as pd
import torch
import torch.nn as nn

import timm


class CustomModel(nn.Module):
    def __init__(self, args=None):
        super(CustomModel, self).__init__()
        #num_classes = 11
        num_classes = 10
        self.backbone = timm.create_model(
            args.backbone_name, pretrained=True, num_classes=num_classes
        )
        # if args.backbone_name == "resnet50":
        #     self.backbone = timm.create_model(
        #         "resnet50", pretrained=True, num_classes=num_classes
        #     )
        # elif args.backbone_name == 'efficientnetb0':
        #     self.backbone = timm.create_model(
        #         "resnet50", pretrained=True, num_classes=num_classes
        #     )

    def forward(self, x):
        x = self.backbone(x)
        return x
