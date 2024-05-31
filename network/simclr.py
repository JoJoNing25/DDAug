import torch
import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
from torch import nn
from PIL import Image
from network.util import get_aspp, get_trunk, make_seg_head, Upsample, initialize_weights, create_circular_mask
from typing import List, Optional, Tuple


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class SimCLR(nn.Module):
    def __init__(self, backbone='resnet-101', projection_dim=128, get_features=True, c_k=7,
                 no_projection_head=False, num_classes=1000):
        super().__init__()
        self.trunk = backbone
        self.get_features = get_features
        self.num_classes = num_classes
        self.backbone, _, _, high_level_ch = get_trunk(self.trunk, imagenet_pretrained=False, 
                                                       c_k=c_k)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if no_projection_head:
            self.projection_head = nn.Identity()
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(high_level_ch, high_level_ch, bias=False),
                nn.BatchNorm1d(high_level_ch),
                nn.ReLU(inplace=True),
                nn.Linear(high_level_ch, projection_dim, bias=False),
                BatchNorm1dNoBias(projection_dim)
            )
        self.high_level_ch = high_level_ch

    def forward(self, x):
        if self.trunk =='unet':
            f = self.backbone(x)
        else:
            _, _, f = self.backbone(x)
        f = self.avgpool(f)
        z = self.projection_head(f.flatten(start_dim=1))
        if self.get_features:
            return f, z
        return z
    
    def adjust_train_mode(self):
        self.eval()
        self.projection_head.train()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = True

    def add_linear_prob(self):
        self.projection_head = nn.Linear(self.high_level_ch, self.num_classes)
        self.projection_head.weight.data.normal_(mean=0.0, std=0.01)
        self.projection_head.bias.data.zero_()
        for _, p in self.named_parameters():
            p.requires_grad = False
        for _, p in self.projection_head.named_parameters():
            p.requires_grad = True
        return self.projection_head
    
    def add_online_cls(self, num_classes=4):
        self.online_cls = nn.Linear(self.high_level_ch, num_classes)
        self.online_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.online_cls.bias.data.zero_()
        for _, p in self.named_parameters():
            p.requires_grad = False
        for _, p in self.online_cls.named_parameters():
            p.requires_grad = True
        return self.online_cls