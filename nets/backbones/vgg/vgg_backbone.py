"""
Created by Wang Han on 2019/7/3 10:21.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn as nn

import nets.backbones.vgg.vgg as vgg


class NormalVggBackbone(nn.Module):
    def __init__(self, orig_vgg):
        super(NormalVggBackbone, self).__init__()

        self.num_features = 512
        self.features = orig_vgg.features

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.features(x)

        return x


class VggBackbone(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, **kwargs):
        arch = self.cfg['network']['backbone']

        if arch == 'vgg16bn':
            orig_vgg = vgg.vgg16_bn(**kwargs)
            arch_net = NormalVggBackbone(orig_vgg)
        else:
            raise Exception('Architecture undefined!')

        return arch_net
