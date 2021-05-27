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


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm_type=None):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', NormSelector.Norm2d(norm_type)(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', NormSelector.Norm2d(norm_type)(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_type=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, norm_type)
            self.add_module('denselayer%d' % (i + 1), layer)


class _TransposeOrInterpolate(nn.Module):
    def __init__(self, num_features, norm_type=None):
        super(_TransposeOrInterpolate, self).__init__()
        self.transpose = nn.Sequential(NormSelector.Norm2d(norm_type)(num_features),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1),
                                       NormSelector.Norm2d(norm_type)(num_features // 2),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(num_features // 2, num_features, kernel_size=2, stride=2))

        self.attention = nn.Sequential(NormSelector.Norm2d(norm_type)(num_features * 2),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(num_features * 2, num_features, kernel_size=1, stride=1),
                                       NormSelector.Norm2d(norm_type)(num_features),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(num_features, num_features * 2, kernel_size=1, stride=1))

    def forward(self, x):
        transpose = self.transpose(x)
        interpolate = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        combine = torch.cat([transpose, interpolate], dim=1)

        attention = self.attention(combine)
        batch, channel, height, width = attention.shape
        attention_ = attention.view(batch, 2, channel // 2, height, width)
        attention = torch.softmax(attention_, dim=1)

        combine = combine.view(batch, 2, channel // 2, height, width)

        select = (combine * attention).sum(dim=1)

        return select


class Up(nn.Module):
    def __init__(self, num_features, num_out, num_layers, drop_rate, bn_size=4, growth_rate=32, norm_type=None,
                 up_type='ua'):
        super(Up, self).__init__()
        self.up_type = up_type
        total_num_feature = num_features[0] + num_features[1]
        self.conv = nn.Sequential(NormSelector.Norm2d(norm_type)(total_num_feature),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(total_num_feature, num_out, kernel_size=1, stride=1))

        self.block = _DenseBlock(num_layers=num_layers, num_input_features=num_out,
                                 bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, norm_type=norm_type)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.block(x)
        return x


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


class UNet(nn.Module):

    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        norm_type = config['network']['norm_type']
        num_classes = config['data']['num_classes']

        self.backbone = BackboneSelector(config).get_backbone()
        num_features_list = self.backbone.get_num_features_list()

        dropout0 = 0.1
        growth_rate = 48
        num_layers = 2
        c_1 = num_features_list[3]
        c_2 = num_features_list[2]
        self.up0 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout0,
                      bn_size=4, growth_rate=growth_rate,
                      norm_type=norm_type)

        c_1 = c_2 + growth_rate * num_layers
        c_2 = num_features_list[1]
        self.up1 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout0,
                      bn_size=4, growth_rate=growth_rate, norm_type=norm_type)

        c_1 = c_2 + growth_rate * num_layers
        c_2 = num_features_list[0]
        self.up2 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout0,
                      bn_size=4, growth_rate=growth_rate, norm_type=norm_type)

        c_1 = c_2 + growth_rate * num_layers
        self.last_conv = nn.Sequential(
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            nn.ReLU(),
            nn.Conv2d(c_1, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        down0, down1, down2, out = self.backbone(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.last_conv(x)

        return x


class MultiViewUNet(nn.Module):

    def __init__(self, config):
        super(MultiViewUNet, self).__init__()
        self.config = config
        num_classes = config['data']['num_classes']
        norm_type = config['network']['norm_type']

        unet_params = config['network']['multi_view_unet']
        dropout = unet_params['dropout']
        num_layers = unet_params['num_layers']
        self.modality_num = unet_params['modality_num']
        self.operator = unet_params['operator']
        if self.operator == 'sum':
            self.sum_weight = unet_params['sum_weight']

        self.backbone = BackboneSelector(config).get_backbone()
        num_features_list = self.backbone.get_num_features_list()
        growth_rate = 48
        c_1 = num_features_list[3]
        c_2 = num_features_list[2]
        self.up0 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout,
                      bn_size=4, growth_rate=growth_rate,
                      norm_type=norm_type)

        c_1 = c_2 + growth_rate * num_layers
        c_2 = num_features_list[1]
        self.up1 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout,
                      bn_size=4, growth_rate=growth_rate, norm_type=norm_type)

        c_1 = c_2 + growth_rate * num_layers
        c_2 = num_features_list[0]
        self.up2 = Up((c_1, c_2), c_2, num_layers=num_layers, drop_rate=dropout,
                      bn_size=4, growth_rate=growth_rate, norm_type=norm_type)
        c_1 = c_2 + growth_rate * num_layers
        self.last_conv = nn.Sequential(
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            _BackwardTransition(c_1, c_1, norm_type=norm_type),
            nn.ReLU(),
            nn.Conv2d(c_1, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        down0, down1, down2, out = self.backbone(x)
        down0 = torch.cat(down0, dim=1) if isinstance(down0, tuple) else down0
        down1 = torch.cat(down1, dim=1) if isinstance(down1, tuple) else down1
        down2 = torch.cat(down2, dim=1) if isinstance(down2, tuple) else down2
        out = torch.cat(out, dim=1) if isinstance(out, tuple) else out

        out = out.view([-1, self.modality_num, out.size(1), out.size(2), out.size(3)])
        if self.operator == 'max':
            out, _ = torch.max(out, dim=1)
        elif self.operator == 'mean':
            out = torch.mean(out, dim=1)
        elif self.operator == 'sum':
            out = self.sum_weight[0] * out[:, 0, ...] + \
                  self.sum_weight[1] * out[:, 1, ...] + \
                  self.sum_weight[2] * out[:, 2, ...]

        down2 = down2.view([-1, self.modality_num, down2.size(1), down2.size(2), down2.size(3)])
        if self.operator == 'max':
            down2, _ = torch.max(down2, dim=1)
        elif self.operator == 'mean':
            down2 = torch.mean(down2, dim=1)
        elif self.operator == 'sum':
            down2 = self.sum_weight[0] * down2[:, 0, ...] + \
                    self.sum_weight[1] * down2[:, 1, ...] + \
                    self.sum_weight[2] * down2[:, 2, ...]
        x = self.up0(out, down2)

        down1 = down1.view([-1, self.modality_num, down1.size(1), down1.size(2), down1.size(3)])
        if self.operator == 'max':
            down1, _ = torch.max(down1, dim=1)
        elif self.operator == 'mean':
            down1 = torch.mean(down1, dim=1)
        elif self.operator == 'sum':
            down1 = self.sum_weight[0] * down1[:, 0, ...] + \
                    self.sum_weight[1] * down1[:, 1, ...] + \
                    self.sum_weight[2] * down1[:, 2, ...]
        x = self.up1(x, down1)

        down0 = down0.view([-1, self.modality_num, down0.size(1), down0.size(2), down0.size(3)])
        if self.operator == 'max':
            down0, _ = torch.max(down0, dim=1)
        elif self.operator == 'mean':
            down0 = torch.mean(down0, dim=1)
        elif self.operator == 'sum':
            down0 = self.sum_weight[0] * down0[:, 0, ...] + \
                    self.sum_weight[1] * down0[:, 1, ...] + \
                    self.sum_weight[2] * down0[:, 2, ...]
        x = self.up2(x, down0)

        return self.last_conv(x)
