import os

import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiSegmentDataset(Dataset):
    def __init__(self, data_root, sample_records, multi_fusion, transforms=None, img_w=512, img_h=512):
        self.data_root = data_root
        self.multi_fusion = multi_fusion
        self.transforms = transforms
        self.img_w, self.img_h = img_w, img_h

        self.records = pd.DataFrame([])
        for sample_record in sample_records:
            self.records = self.records.append(pd.read_csv(sample_record, dtype='str'))
        print(">>> The number of records is {}".format(len(self.records)))

    def _resolve_record(self, record):
        base_ct_path = os.path.join(self.data_root, record.base_ct_path)
        pre_ct_path = os.path.join(self.data_root, record.pre_ct_path)
        next_ct_path = os.path.join(self.data_root, record.next_ct_path)
        target_path = os.path.join(self.data_root, record.target_path)

        base_ct = sitk.GetArrayFromImage(sitk.ReadImage(base_ct_path))[0]
        try:
            pre_ct = sitk.GetArrayFromImage(sitk.ReadImage(pre_ct_path))[0]
        except:
            pre_ct = -5000 * np.ones([self.img_w, self.img_h])
        try:
            next_ct = sitk.GetArrayFromImage(sitk.ReadImage(next_ct_path))[0]
        except:
            next_ct = -5000 * np.ones([self.img_w, self.img_h])

        try:
            mask = cv2.imread(target_path)[:, :, 0]
            np.place(mask, mask == 255, 1)
        except:
            mask = np.zeros((self.img_w, self.img_h), dtype=np.uint8)

        imgs = [base_ct, pre_ct, next_ct]

        return imgs, mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        imgs, mask = self._resolve_record(self.records.iloc[idx])
        sample = {'image': imgs, 'label': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        if not self.multi_fusion:
            # 将3输入转化成3通道
            sample['image'] = torch.transpose(sample['image'], 0, 1).squeeze()

        return sample
