"""
Created by Wang Han on 2019/5/8 11:27.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
"""
from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.resnet import resnet


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        expansion = orig_resnet.expansion
        self.num_features_list = [64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]

        # take pretrained resnet, except AvgPool and FC
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.prefix(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features


class OriUNet(nn.Module):
    """
    Original UNet with ResNet50 as its backbone.
    """

    def __init__(self, config):
        super(OriUNet, self).__init__()
        self.config = config
        color_channels = self.config['data']['color_channels']
        norm_type = config['network']['norm_type']
        num_classes = config['data']['num_classes']

        # encoder
        orig_resnet = resnet.resnet50(color_channels=color_channels, norm_type=norm_type)
        self.backbone = NormalResnetBackbone(orig_resnet)
        num_features_list = self.backbone.get_num_features_list()

        self.up1 = Up(num_features_list[4] + num_features_list[3], num_features_list[3])
        self.up2 = Up(num_features_list[3] + num_features_list[2], num_features_list[2])
        self.up3 = Up(num_features_list[2] + num_features_list[1], num_features_list[1])
        self.up4 = Up(num_features_list[1] + num_features_list[0], num_features_list[0])
        self.outc = OutConv(num_features_list[0], num_classes)

    def forward(self, x):
        # x1.shape [8, 64, 144, 144]
        # x2.shape [8, 256, 72, 72]
        # x3.shape [8, 512, 36, 36]
        # x4.shape [8, 1024, 18, 18]
        # x5.shape [8, 2048, 9, 9]
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # due to the downsampling in ResUNet's first layer
        logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=True)
        return logits
