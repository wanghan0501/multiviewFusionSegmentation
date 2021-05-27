import os

import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(self, data_root, sample_records, transforms=None, img_w=512, img_h=512):
        self.data_root = data_root
        self.transforms = transforms
        self.img_w, self.img_h = img_w, img_h

        self.records = pd.DataFrame([])
        for sample_record in sample_records:
            self.records = self.records.append(pd.read_csv(sample_record, dtype='str'))
        print(">>> The number of records is {}".format(len(self.records)))

    def _resolve_record(self, record):
        # ct
        base_ct_path = os.path.join(self.data_root, record.base_ct_path)
        base_ct = sitk.GetArrayFromImage(sitk.ReadImage(base_ct_path))[0]
        # mask
        target_path = os.path.join(self.data_root, record.target_path)
        try:
            mask = cv2.imread(target_path)[:, :, 0]
            np.place(mask, mask == 255, 1)
        except:
            mask = np.zeros((self.img_w, self.img_h), dtype=np.uint8)

        return base_ct, mask

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img, mask = self._resolve_record(self.records.iloc[idx])
        sample = {'image': img, 'label': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
