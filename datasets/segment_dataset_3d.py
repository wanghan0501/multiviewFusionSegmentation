"""
Created by Wang Han on 2019/6/13 12:41.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SegmentDataset3D(Dataset):
    def __init__(self, data_root, sample_records, transforms=None):
        self.data_root = data_root
        self.records = pd.DataFrame([])
        for sample_record in sample_records:
            self.records = self.records.append(pd.read_csv(sample_record, dtype='str'))
        self.transforms = transforms
        print(">>> The number of records is {}".format(len(self.records)))

    def _resolve_record(self, record):
        CT_path = os.path.join(self.data_root, record.CT_path)
        mask_path = os.path.join(self.data_root, record.mask_path)

        img = np.load(CT_path)
        mask = np.load(mask_path)
        return img, mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img, mask = self._resolve_record(self.records.iloc[idx])
        sample = {'image': img, 'label': mask}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample['original_size'] = img.shape
        return sample
