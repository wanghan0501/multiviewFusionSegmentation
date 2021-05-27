"""
Created by Wang Han on 2019/3/30 13:18.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn as nn

import nets.backbones.resnet3d.resnet3d as resnet3d


class NormalResnet3DBackbone(nn.Module):
    def __init__(self, orig_resnet3d):
        super(NormalResnet3DBackbone, self).__init__()

        expansion = orig_resnet3d.expansion
        self.num_features = 512 * expansion
        self.num_features_list = [64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet3d.conv1
        self.bn1 = orig_resnet3d.bn1
        self.relu = orig_resnet3d.relu
        self.maxpool = orig_resnet3d.maxpool
        self.layer1 = orig_resnet3d.layer1
        self.layer2 = orig_resnet3d.layer2
        self.layer3 = orig_resnet3d.layer3
        self.layer4 = orig_resnet3d.layer4

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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


class DilatedResnet3DBackbone(nn.Module):
    def __init__(self, orig_resnet3d, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnet3DBackbone, self).__init__()

        expansion = orig_resnet3d.expansion
        self.num_features = 512 * expansion
        self.num_features_list = [64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
        from functools import partial

        if dilate_scale == 8:
            orig_resnet3d.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet3d.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet3d.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet3d.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet3d.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet3d.conv1
        self.bn1 = orig_resnet3d.bn1
        self.relu = orig_resnet3d.relu
        self.maxpool = orig_resnet3d.maxpool
        self.layer1 = orig_resnet3d.layer1
        self.layer2 = orig_resnet3d.layer2
        self.layer3 = orig_resnet3d.layer3
        self.layer4 = orig_resnet3d.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            # the convolution with stride
            if m.stride == (2, 2, 2):
                m.stride = (1, 1, 1)
                if m.kernel_size == (3, 3, 3):
                    # m.dilation = (dilate // 2, dilate // 2, dilate // 2)
                    # m.padding = (dilate // 2, dilate // 2, dilate // 2)
                    m.dilation = (dilate, dilate, dilate)
                    m.padding = (dilate, dilate, dilate)
            else:
                if m.kernel_size == (3, 3, 3):
                    m.dilation = (dilate, dilate, dilate)
                    m.padding = (dilate, dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
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


class ResNet3DBackbone(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, **kwargs):
        arch = self.cfg['network']['backbone']

        if arch == 'resnet3d10':
            orig_resnet = resnet3d.resnet10(**kwargs)
            arch_net = NormalResnet3DBackbone(orig_resnet)
        elif arch == 'resnet3d10_dilated8':
            orig_resnet = resnet3d.resnet10(**kwargs)
            arch_net = DilatedResnet3DBackbone(orig_resnet, dilate_scale=8, multi_grid=None)
        elif arch == 'resnet3d18':
            orig_resnet = resnet3d.resnet18(**kwargs)
            arch_net = NormalResnet3DBackbone(orig_resnet)
        elif arch == 'resnet3d18_dilated8':
            orig_resnet = resnet3d.resnet18(**kwargs)
            arch_net = DilatedResnet3DBackbone(orig_resnet, dilate_scale=8, multi_grid=None)
        elif arch == 'resnet3d34':
            orig_resnet = resnet3d.resnet34(**kwargs)
            arch_net = NormalResnet3DBackbone(orig_resnet)
        elif arch == 'resnet3d34_dilated8':
            orig_resnet = resnet3d.resnet34(**kwargs)
            arch_net = DilatedResnet3DBackbone(orig_resnet, dilate_scale=8, multi_grid=None)
        elif arch == 'resnet3d50':
            orig_resnet = resnet3d.resnet50(**kwargs)
            arch_net = NormalResnet3DBackbone(orig_resnet)
        elif arch == 'resnet3d50_dilated8':
            orig_resnet = resnet3d.resnet50(**kwargs)
            arch_net = DilatedResnet3DBackbone(orig_resnet, dilate_scale=8, multi_grid=None)
        elif arch == 'resnet3d101':
            orig_resnet = resnet3d.resnet101(**kwargs)
            arch_net = NormalResnet3DBackbone(orig_resnet)
        elif arch == 'resnet3d101_dilated8':
            orig_resnet = resnet3d.resnet101(**kwargs)
            arch_net = DilatedResnet3DBackbone(orig_resnet, dilate_scale=8, multi_grid=None)
        else:
            raise Exception('Architecture undefined!')

        return arch_net
