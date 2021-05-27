"""
Created by Wang Han on 2019/3/30 14:05.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.backbone_selector import BackboneSelector
from nets.cores.norm_selector import NormSelector


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, inner_features=512, out_features=512, rate=1, norm_type=None):
        super(ASPPModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(inner_features, out_features, 1, 1, padding=0, dilation=rate, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(inner_features, out_features, 3, 1, padding=6 * rate, dilation=6 * rate, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(inner_features, out_features, 3, 1, padding=12 * rate, dilation=12 * rate, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(inner_features, out_features, 3, 1, padding=18 * rate, dilation=18 * rate, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True))
        self.branch_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inner_features, out_features, 1, 1, padding=0, dilation=1, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, 1, 1, padding=0, dilation=1, bias=False),
            NormSelector.Norm2d(norm_type)(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        global_feats = F.interpolate(self.branch_pooling(x), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat((feat1, feat2, feat3, feat4, global_feats), 1)

        bottle = self.bottleneck(out)
        return bottle


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()
        self.cfg = cfg
        norm_type = cfg['network']['norm_type']
        num_classes = cfg['data']['num_classes']

        self.backbone = BackboneSelector(cfg).get_backbone()
        self.aspp = ASPPModule(2048, 512, norm_type=norm_type)

        self.shallow_conv = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1, bias=False),
            NormSelector.Norm2d(norm_type)(512),
            nn.ReLU(inplace=True))
        self.cat_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, padding=1, bias=False),
            NormSelector.Norm2d(norm_type)(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0))

        # self.cls_conv = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        #     NormSelector.Norm2d(norm_type)(256),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):

        # from time import time
        # start = time()
        x = self.backbone(x)
        # middle = time()
        aspp_feature = self.aspp(x[-1])
        # x_cls = self.cls_conv(aspp_feature)

        upsample_aspp_feature = F.interpolate(aspp_feature, scale_factor=4, mode="bilinear", align_corners=True)
        shallow_feature = self.shallow_conv(x[-4])
        cat_feature = torch.cat([upsample_aspp_feature, shallow_feature], 1)
        x = self.cat_conv(cat_feature)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        # end = time()
        # print("backbone: {}, aspp: {}".format(middle - start, end - start))
        return x
