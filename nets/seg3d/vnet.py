"""
Created by Wang Han on 2019/6/13 12:49.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, padding=1, dropout_rate=0):
        super(DownSampling, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_rate > 0:
            out = F.dropout3d(out)
        return out


class EncoderBlock(nn.Module):
    '''
    Encoder block
    '''

    def __init__(self, in_channels, out_channels, stride=1, padding=1, num_groups=8):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.actv1 = nn.ReLU(inplace=True)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=padding)
        self.actv2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out


class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''

    def __init__(self, in_channels, out_channels, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x, skip_x=None):
        out = self.conv1(x)
        if skip_x is not None:
            out = F.interpolate(out, skip_x.shape[2:], mode=self.mode, align_corners=self.align_corners)
            out = torch.cat((out, skip_x), 1)
            out = self.conv2(out)
        else:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return out


class UpSamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)

        return out


class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''

    def __init__(self, in_channels, out_channels, stride=1, padding=1, num_groups=8, dropout_rate=0):
        super(DecoderBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.actv1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.actv2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        if self.dropout_rate > 0:
            out = F.dropout3d(out)

        return out


class VNet(nn.Module):
    def __init__(self, config):
        super(VNet, self).__init__()

        self.config = config
        # some critical parameters

        color_channels = config['data']['color_channels']
        num_classes = config['data']['num_classes']

        # Encoder Blocks
        self.in_conv0 = DownSampling(color_channels, 32, stride=1)
        self.en_block0 = EncoderBlock(32, 32)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64)
        self.en_block1_1 = EncoderBlock(64, 64)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128)
        self.en_block2_1 = EncoderBlock(128, 128)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256)
        self.en_block3_1 = EncoderBlock(256, 256)
        self.en_block3_2 = EncoderBlock(256, 256)
        self.en_block3_3 = EncoderBlock(256, 256)

        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128)
        self.de_block2 = DecoderBlock(128, 128)
        self.de_up1 = LinearUpSampling(128, 64)
        self.de_block1 = DecoderBlock(64, 64)
        self.de_up0 = LinearUpSampling(64, 32)
        self.de_block0 = DecoderBlock(32, 32)
        self.de_end = nn.Conv3d(32, num_classes, kernel_size=1)

        # Initialise weights
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias)

    def forward(self, x):
        out_init = self.in_conv0(x)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))

        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        out_end = self.de_end(out_de0)

        return out_end
