"""
Created by Wang Han on 2019/3/28 22:00.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn as nn

import nets.backbones.densenet.densenet as densenet


class NormalDensenetBackbone(nn.Module):
    def __init__(self, orig_densenet):
        super(NormalDensenetBackbone, self).__init__()

        self.num_features = orig_densenet.num_features
        self.num_features_list = orig_densenet.num_features_list
        # take pretrained resnet, except AvgPool and FC
        self.conv0 = orig_densenet.features.conv0
        self.norm0 = orig_densenet.features.norm0
        self.relu0 = orig_densenet.features.relu0
        self.pool0 = orig_densenet.features.pool0

        self.denseblock1 = orig_densenet.features.denseblock1
        self.transition1 = orig_densenet.features.transition1
        self.transition1_pool = orig_densenet.features.transition1_pool

        self.denseblock2 = orig_densenet.features.denseblock2
        self.transition2 = orig_densenet.features.transition2
        self.transition2_pool = orig_densenet.features.transition2_pool

        self.denseblock3 = orig_densenet.features.denseblock3
        self.transition3 = orig_densenet.features.transition3
        self.transition3_pool = orig_densenet.features.transition3_pool

        self.denseblock4 = orig_densenet.features.denseblock4

        self.norm5 = orig_densenet.features.norm5

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        x = self.transition1(x)
        tuple_features.append(x)
        x = self.transition1_pool(x)

        x = self.denseblock2(x)
        x = self.transition2(x)
        tuple_features.append(x)
        x = self.transition2_pool(x)

        x = self.denseblock3(x)
        x = self.transition3(x)
        tuple_features.append(x)
        x = self.transition3_pool(x)

        x = self.denseblock4(x)

        x = self.norm5(x)
        tuple_features.append(x)

        return tuple_features


class DilatedDensenetBackbone(nn.Module):
    def __init__(self, orig_densenet, dilate_scale=8):
        super(DilatedDensenetBackbone, self).__init__()

        self.num_features = orig_densenet.num_features
        self.num_features_list = orig_densenet.num_features_list
        self.dilate_scale = dilate_scale
        from functools import partial

        if dilate_scale == 8:
            orig_densenet.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))
            orig_densenet.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))

        elif dilate_scale == 16:
            orig_densenet.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))

        self.conv0 = orig_densenet.features.conv0
        self.norm0 = orig_densenet.features.norm0
        self.relu0 = orig_densenet.features.relu0
        self.pool0 = orig_densenet.features.pool0

        self.denseblock1 = orig_densenet.features.denseblock1
        self.transition1 = orig_densenet.features.transition1
        self.transition1_pool = orig_densenet.features.transition1_pool

        self.denseblock2 = orig_densenet.features.denseblock2
        self.transition2 = orig_densenet.features.transition2
        self.transition2_pool = orig_densenet.features.transition2_pool

        self.denseblock3 = orig_densenet.features.denseblock3
        self.transition3 = orig_densenet.features.transition3
        self.transition3_pool = orig_densenet.features.transition3_pool

        self.denseblock4 = orig_densenet.features.denseblock4

        self.norm5 = orig_densenet.features.norm5

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        x = self.transition1(x)
        tuple_features.append(x)
        x = self.transition1_pool(x)

        x = self.denseblock2(x)
        x = self.transition2(x)
        tuple_features.append(x)
        if self.dilate_scale > 8:
            x = self.transition2_pool(x)

        x = self.denseblock3(x)
        x = self.transition3(x)
        tuple_features.append(x)
        if self.dilate_scale > 16:
            x = self.transition3_pool(x)

        x = self.denseblock4(x)

        x = self.norm5(x)
        tuple_features.append(x)

        return tuple_features


class DenseNetBackbone(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, **kwargs):
        arch = self.cfg['network']['backbone']
        if arch == 'densenet121':
            orig_densenet = densenet.densenet121(**kwargs)
            arch_net = NormalDensenetBackbone(orig_densenet)

        elif arch == 'densenet121_dilated8':
            orig_densenet = densenet.densenet121(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=8)

        elif arch == 'densenet121_dilated16':
            orig_densenet = densenet.densenet121(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=16)

        elif arch == 'densenet169':
            orig_densenet = densenet.densenet169(**kwargs)
            arch_net = NormalDensenetBackbone(orig_densenet)

        elif arch == 'densenet169_dilated8':
            orig_densenet = densenet.densenet169(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=8)

        elif arch == 'densenet169_dilated16':
            orig_densenet = densenet.densenet169(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=16)

        elif arch == 'densenet201':
            orig_densenet = densenet.densenet201(**kwargs)
            arch_net = NormalDensenetBackbone(orig_densenet)

        elif arch == 'densenet201_dilated8':
            orig_densenet = densenet.densenet201(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=8)

        elif arch == 'densenet201_dilated16':
            orig_densenet = densenet.densenet201(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=16)

        elif arch == 'densenet161':
            orig_densenet = densenet.densenet161(**kwargs)
            arch_net = NormalDensenetBackbone(orig_densenet)

        elif arch == 'densenet161_dilated8':
            orig_densenet = densenet.densenet161(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=8)

        elif arch == 'densenet161_dilated16':
            orig_densenet = densenet.densenet161(**kwargs)
            arch_net = DilatedDensenetBackbone(orig_densenet, dilate_scale=16)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
