"""
Created by Wang Han on 2019/3/28 22:10.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

from nets.backbones.densenet.densenet_backbone import DenseNetBackbone
from nets.backbones.resnet.resnet_backbone import ResNetBackbone
from nets.backbones.resnet3d.resnet3d_backbone import ResNet3DBackbone
from nets.backbones.vgg.vgg_backbone import VggBackbone


class BackboneSelector(object):

    def __init__(self, config):
        self.config = config

    def get_backbone(self):
        color_channels = self.config['data']['color_channels']
        backbone = self.config['network']['backbone']
        norm_type = self.config['network']['norm_type']

        if 'resnet3d' in backbone:
            model = ResNet3DBackbone(self.config)(color_channels=color_channels)
        elif 'resnet' in backbone:
            model = ResNetBackbone(self.config)(color_channels=color_channels, norm_type=norm_type)
        elif 'densenet' in backbone:
            model = DenseNetBackbone(self.config)(color_channels=color_channels, norm_type=norm_type)
        elif 'vgg' in backbone:
            model = VggBackbone(self.config)(color_channels=color_channels)
        else:
            raise Exception('Not support backbone: {}.'.format(backbone))
        return model
