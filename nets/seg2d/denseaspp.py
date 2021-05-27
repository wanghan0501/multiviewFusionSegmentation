"""
Created by Wang Han on 2019/3/28 21:59.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.backbones.backbone_selector import BackboneSelector
from nets.cores.norm_selector import NormSelector


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """

    def __init__(self, config):
        super(DenseASPP, self).__init__()
        self.config = config
        norm_type = config['network']['norm_type']
        num_classes = config['data']['num_classes']
        dropout0 = 0.1
        dropout1 = 0.1

        self.backbone = BackboneSelector(config).get_backbone()

        num_features = self.backbone.get_num_features()

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=256, num2=64,
                                      dilation_rate=3, drop_out=dropout0, norm_type=norm_type)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + 64 * 1, num1=256, num2=64,
                                      dilation_rate=6, drop_out=dropout0, norm_type=norm_type)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + 64 * 2, num1=256, num2=64,
                                       dilation_rate=12, drop_out=dropout0, norm_type=norm_type)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + 64 * 3, num1=256, num2=64,
                                       dilation_rate=18, drop_out=dropout0, norm_type=norm_type)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + 64 * 4, num1=256, num2=64,
                                       dilation_rate=24, drop_out=dropout0, norm_type=norm_type)

        num_features = num_features + 5 * 64

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features,
                      out_channels=num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        feature = self.backbone(x)[-1]
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        cls = self.classification(feature)

        out = F.interpolate(cls, scale_factor=8, mode='bilinear', align_corners=False)

        return out


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, norm_type):
        super(_DenseAsppBlock, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm1', NormSelector.Norm2d(norm_type)(num_features=num1)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                           dilation=dilation_rate, padding=dilation_rate)),
        self.add_module('norm2', NormSelector.Norm2d(norm_type)(num_features=num2)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_type=None):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('norm', NormSelector.Norm2d(norm_type)(num_features=num_output_features)),
