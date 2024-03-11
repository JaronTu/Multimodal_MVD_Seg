import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
# from nnunetv2.paths import nnUNet_raw_data
import SimpleITK as sitk
import os
import shutil
from typing import Union

def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    # print(uniques)
    for u in uniques:
        if u not in [0, 1, 2, 3, 4, 5]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    # 1-brainstem 2-artery 3/4-nerve 5-vein
    seg_new[img_npy == 1] = 1
    seg_new[img_npy == 2] = 2
    seg_new[img_npy == 3] = 2
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 5] = 3
    # seg_new[img_npy == 1] = 1
    # seg_new[img_npy == 2] = 2
    # seg_new[img_npy == 3] = 4
    # seg_new[img_npy == 4] = 4
    # seg_new[img_npy == 5] = 3
    # 49
    seg_new[img_npy == 1] = 1
    seg_new[img_npy == 2] = 2
    seg_new[img_npy == 3] = 3
    seg_new[img_npy == 4] = 4
    seg_new[img_npy == 5] = 4
    # img_corr = sitk.GetImageFromArray(seg_new)
    img_corr = sitk.GetImageFromArray(img_npy)
    # img_corr.CopyInformation(img)
    # exit()
    sitk.WriteImage(img_corr, out_file)
    # sitk.WriteImage(img_corr, in_file)

if __name__ == '__main__':
    in_dir = '/home/hci/turzh/Datasets/nerve_vessel/usable_dataset'
    # out_dir = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset333_nervessel/labelsTr'
    out_dir = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset334_nervessel/labelsTr'
    list1 = ['74_gt','78_gt','84_gt','85_gt','86_gt','87_gt','88_gt','89_gt','90_gt','91_gt','92_gt','93_gt', \
             '94_gt', '95_gt', '96_gt', '97_gt', '99_gt', '100_gt'] #3-静脉 4，5-神经
    list2 = ['05_gt', '08_gt', '13_gt', '17_gt', '19_gt', '22_gt', '28_gt', '34_gt', '36_gt', '37_gt', '38_gt', \
             '39_gt', '53_gt'] #3-4 4-3
    list3 = ['24_gt', '29_gt']
    list4 = ['49_gt']
    list5 = ['72_gt']
    # list1 = ['003_gt','006_gt']
    # if any(e in data for e in list1):
    # result = [x for x in list1 if x not in list2]
    # list_all = list(set(list1).union(list2, list3, list4, list5))
    for data in os.listdir(in_dir):
        # print(data[-12:-7])
        if data.endswith('_gt.nii.gz') and data[-12:-7] in list4:
            file = os.path.join(in_dir, data)
            out_file = os.path.join(out_dir, data.split('_')[0]+'.nii.gz')
            print(out_file)
            # exit()
            copy_BraTS_segmentation_and_convert_labels(file, out_file)

    # file = '/home/hci/turzh/Datasets/nerve_vessel/usable_dataset/pat072_gt.nii.gz'
    # out_file = '/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task333_nervessel/labelsTr/pat072.nii.gz'
    # copy_BraTS_segmentation_and_convert_labels(file, out_file)

