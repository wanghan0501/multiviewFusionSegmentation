"""
Created by Wang Han on 2020/6/27 22:14.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""
import numbers
from random import random

import numpy as np
import torch
from scipy.ndimage import zoom


class DropInvalidRange:
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # 已左上角的值为参考
        zero_value = img[0, 0, 0]
        non_zeros_idx = np.where(img != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if mask is not None:
            sample['image'] = img[min_z:max_z, min_h:max_h, min_w:max_w]
            sample['label'] = mask[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            sample['image'] = img[min_z:max_z, min_h:max_h, min_w:max_w]
        return sample


class RandomCenterCrop:
    """
    Random crop
    """

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        target_indexs = np.where(mask > 0)
        [img_d, img_h, img_w] = img.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        sample['image'] = img[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
        sample['label'] = mask[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
        return sample


class ItensityNormalize:
    def __call__(self, sample):
        img = sample['image']
        pixels = img[img > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (img - mean) / std
        out_random = np.random.normal(0, 1, size=img.shape)
        out[img == 0] = out_random[img == 0]
        sample['image'] = img
        return sample


class Normalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        img = (img - self.mean) / self.std
        sample['image'] = img
        return sample


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        scale = np.array(self.size) / np.array(img.shape)
        img = zoom(img, scale, order=1)
        mask = zoom(mask, scale, order=1)
        sample['image'] = img
        sample['label'] = mask
        return sample


class ToTensor:
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        image = np.expand_dims(img, 0)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        sample['image'] = image
        sample['label'] = mask
        return sample
