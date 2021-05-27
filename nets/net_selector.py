"""
Created by Wang Han on 2019/5/13 15:21.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
from nets.seg2d.ddcnn import DDCNN
from nets.seg2d.deeplabv3plus import DeepLabV3Plus
from nets.seg2d.denseaspp import DenseASPP
from nets.seg2d.multiviewunet import UNet, MultiViewUNet
from nets.seg2d.ori_unet import OriUNet
from nets.seg2d.res_unet import ResUNet
from nets.seg3d.vnet import VNet
from nets.seg3d.voxresnet import VoxResNet


class NetSelector(object):

    def __init__(self, config):
        self.config = config

    def get_net(self):
        net_name = self.config['network']['net_name']

        if net_name == 'denseaspp':
            net = DenseASPP(self.config)
        elif net_name == 'deeplabv3plus':
            net = DeepLabV3Plus(self.config)
        elif net_name == 'ori_unet':
            net = OriUNet(self.config)
        elif net_name == 'unet':
            net = UNet(self.config)
        elif net_name == 'multi_view_unet':
            net = MultiViewUNet(self.config)
        elif net_name == 'resunet':
            net = ResUNet(self.config)
        elif net_name == 'voxresnet':
            net = VoxResNet(self.config)
        elif net_name == 'vnet':
            net = VNet(self.config)
        elif net_name == 'ddcnn':
            net = DDCNN(self.config)
        else:
            raise Exception('Not support net: {}.'.format(net_name))

        return net
