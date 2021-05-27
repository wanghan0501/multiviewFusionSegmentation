"""
Created by Wang Han on 2019/5/8 11:27.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.backbone_selector import BackboneSelector
from nets.cores.norm_selector import NormSelector


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None):
        super(BasicBlock, self).__init__()
        self.norm1 = NormSelector.Norm2d(norm_type)(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm2 = NormSelector.Norm2d(norm_type)(planes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out_preact = self.relu(out)
        out = self.conv1(out_preact)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(out_preact)

        out += residual

        return out


class _BackwardTransition(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=None):
        super(_BackwardTransition, self).__init__()
        self.conv = nn.Sequential(
            NormSelector.Norm2d(norm_type)(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)
        return x


class Up(nn.Module):
    def __init__(self, num_features, num_out, norm_type=None,
                 up_type='ua'):
        super(Up, self).__init__()
        self.up_type = up_type
        total_num_feature = num_features[0] + num_features[1]
        self.conv = nn.Sequential(NormSelector.Norm2d(norm_type)(total_num_feature),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(total_num_feature, num_out, kernel_size=1, stride=1))

        self.block = BasicBlock(inplanes=num_out, planes=num_out, norm_type=norm_type)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.block(x)
        return x


class ResUNet(nn.Module):

    def __init__(self, config):
        super(ResUNet, self).__init__()
        self.config = config
        norm_type = config['network']['norm_type']
        num_classes = config['data']['num_classes']

        self.backbone = BackboneSelector(config).get_backbone()
        num_features_list = self.backbone.get_num_features_list()

        self.drop_rate = 0.1
        c_1, c_2 = num_features_list[3], num_features_list[2]

        self.up0 = Up((c_1, c_2), c_2, norm_type=norm_type)

        c_1 = c_2
        c_2 = num_features_list[1]
        self.up1 = Up((c_1, c_2), c_2, norm_type=norm_type)

        c_1 = c_2
        c_2 = num_features_list[0]
        self.up2 = Up((c_1, c_2), c_2, norm_type=norm_type)

        c_1 = c_2
        self.last_conv = nn.Sequential(
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            nn.ReLU(),
            nn.Conv2d(c_1, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        down0, down1, down2, down3 = self.backbone(x)
        x = self.up0(down3, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_conv(x)

        return x
