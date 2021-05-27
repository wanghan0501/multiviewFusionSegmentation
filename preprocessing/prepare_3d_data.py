import sys

from tqdm import tqdm

sys.path.append('..')

import operator
import os

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import SimpleITK as sitk
import torch
from torchvision.transforms import Compose

from datasets.transforms import segment_transforms as st


def get_slice_order(sample_path, modality="CT"):
    slice_dict = {}
    for filename in os.listdir(sample_path):
        if filename.startswith(modality):
            key = filename

            s = dicom.read_file(os.path.join(sample_path, filename),
                                force=True)
            value = s.ImagePositionPatient[-1]
            slice_dict[key] = value
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def perpare_3d_data(data_root, record):
    sample_records = pd.read_csv(record, dtype='str', names=['sample_path', 'tumor_type'])
    trans = Compose([
        st.FilterAndNormalizeByWL(250, 50),
        st.Arr2Image(),
        st.CenterCrop([288, 288]),
        st.ToTensor()
    ])
    npy_records = []
    for idx, row in tqdm(sample_records.iterrows()):
        sample_path = os.path.join(data_root, row.sample_path)
        samples = get_slice_order(sample_path)
        cts = []
        masks = []
        for idx, sample in enumerate(samples):
            basename = sample[0].strip('.dcm')
            base_ct_path = os.path.abspath(os.path.join(sample_path, sample[0]))
            # check pinnacle_data or raystation_data
            if "pinnacle_data" in base_ct_path:  # pinnacle_data
                mask_path = os.path.abspath(
                    os.path.join(sample_path, 'mask/GTVtb', '{}.bmp'.format(basename)))
                patient_id = sample_path.split('/')[-2]
            else:  # raystation_data
                mask_path = os.path.abspath(
                    os.path.join(sample_path, 'mask/GTVtb', '{}.bmp'.format(basename.strip('CT.'))))
                patient_id = sample_path.split('/')[-1]

            if os.path.exists(mask_path):
                img = sitk.GetArrayFromImage(sitk.ReadImage(base_ct_path))[0]
                mask = cv2.imread(mask_path)[:, :, 0]
                np.place(mask, mask == 255, 1)
                com = {'image': img, 'label': mask}
                com = trans(com)

                cts.append(com['image'])
                masks.append(com['label'])
        cts = np.stack(cts, axis=1).squeeze()
        masks = np.stack(masks, axis=0)
        cts_saved_name = 'image/CT_{}.npy'.format(patient_id)
        cts_save_path = '{}/{}'.format(npy_saved_path, cts_saved_name)
        masks_save_name = 'mask/M_{}.npy'.format(patient_id)
        masks_save_path = '{}/{}'.format(npy_saved_path, masks_save_name)
        np.save(cts_save_path, cts)
        np.save(masks_save_path, masks)
        npy_records.append([cts_saved_name, masks_save_name])
    npy_df = pd.DataFrame(npy_records, columns=['CT_path', 'mask_path'])
    return npy_df


if __name__ == '__main__':
    data_root = '/root/workspace/DeepRadiology/target_segmentation/'
    record_read_path = f'{data_root}/records/胶质恶性细胞瘤/new_paper/'
    npy_saved_path = f'{data_root}/3d/npy_data'
    record_saved_path = f'{data_root}/3d/records/'

    if not os.path.exists(npy_saved_path):
        os.makedirs(npy_saved_path)
        os.makedirs(f'{npy_saved_path}/image')
        os.makedirs(f'{npy_saved_path}/mask')

    if not os.path.exists(record_saved_path):
        os.makedirs(record_saved_path)

    train_3d_df = perpare_3d_data(data_root, f'{record_read_path}/train_samples.csv')
    train_3d_df.to_csv(f'{record_saved_path}/3d_train_samples.csv', index=False)

    eval_3d_df = perpare_3d_data(data_root, f'{record_read_path}/eval_samples.csv')
    eval_3d_df.to_csv(f'{record_saved_path}/3d_eval_samples.csv', index=False)

    test_3d_df = perpare_3d_data(data_root, f'{record_read_path}/test_samples.csv')
    test_3d_df.to_csv(f'{record_saved_path}/3d_test_samples.csv', index=False)
