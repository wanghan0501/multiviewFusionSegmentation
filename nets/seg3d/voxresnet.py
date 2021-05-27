"""
Created by Wang Han on 2019/9/25 22:47.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.backbone_selector import BackboneSelector
from nets.backbones.resnet3d.resnet3d import BasicBlock


class Up(nn.Module):
    def __init__(self, num_features, num_out):
        super(Up, self).__init__()
        total_num_feature = num_features[0] + num_features[1]
        self.conv = nn.Sequential(nn.BatchNorm3d(total_num_feature),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(total_num_feature, num_out, kernel_size=1, stride=1))

        self.block = BasicBlock(inplanes=num_out, planes=num_out)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.block(x)
        return x


class VoxResNet(nn.Module):

    def __init__(self, config):
        super(VoxResNet, self).__init__()
        self.config = config
        num_classes = config['data']['num_classes']

        self.backbone = BackboneSelector(config).get_backbone()
        num_features_list = self.backbone.get_num_features_list()

        c_1, c_2 = num_features_list[3], num_features_list[2]
        self.up0 = Up((c_1, c_2), c_2)

        c_1 = c_2
        c_2 = num_features_list[1]
        self.up1 = Up((c_1, c_2), c_2)

        c_1 = c_2
        c_2 = num_features_list[0]
        self.up2 = Up((c_1, c_2), c_2)

        c_1 = c_2
        self.conv_seg = nn.Sequential(
            nn.Conv3d(c_1, c_1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(c_1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_1, num_classes, kernel_size=1, stride=1, padding=0))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        d, h, w = x.size()[2:]
        down0, down1, down2, down3 = self.backbone(x)
        x = self.up0(down3, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.conv_seg(x)
        x = F.interpolate(x, size=(d, h, w), mode='trilinear', align_corners=True)

        return x
