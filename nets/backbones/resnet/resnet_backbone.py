"""
Created by Wang Han on 2019/3/30 13:18.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import torch.nn as nn

import nets.backbones.resnet.resnet as resnet

class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        # self.num_features = 2048
        # self.num_features_list = [256, 512, 1024, 2048]
        expansion = orig_resnet.expansion
        self.num_features = 512 * expansion
        self.num_features_list = [64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]

        # take pretrained resnet, except AvgPool and FC
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.prefix(x)
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


class DilatedResnetBackbone(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnetBackbone, self).__init__()

        self.num_features = 2048
        self.num_features_list = [256, 512, 1024, 2048]
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def get_num_features_list(self):
        return self.num_features_list

    def forward(self, x):
        tuple_features = list()
        x = self.prefix(x)
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


class ResNetBackbone(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, **kwargs):
        arch = self.cfg['network']['backbone']

        if arch == 'resnet34':
            orig_resnet = resnet.resnet34(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512

        elif arch == 'resnet34_dilated8':
            orig_resnet = resnet.resnet34(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8)
            arch_net.num_features = 512

        elif arch == 'resnet34_dilated16':
            orig_resnet = resnet.resnet34(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16)
            arch_net.num_features = 512

        elif arch == 'resnet50':
            orig_resnet = resnet.resnet50(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8)

        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16)

        elif arch == 'deepbase_resnet50':
            orig_resnet = resnet.deepbase_resnet50(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'deepbase_resnet50_dilated8':
            orig_resnet = resnet.deepbase_resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8)

        elif arch == 'deepbase_resnet50_dilated16':
            orig_resnet = resnet.deepbase_resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16)

        elif arch == 'resnet101':
            orig_resnet = resnet.resnet101(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.resnet101(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8)

        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.resnet101(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16)
        else:
            raise Exception('Architecture undefined!')

        return arch_net
