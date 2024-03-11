import torch
import os
import SimpleITK as sitk
import numpy as np

path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/label/pat113.nii.gz'
label = sitk.ReadImage(path)
label = sitk.GetArrayFromImage(label)
print(label.shape)
# label_ = label[:200, :, :]
label_ = label[-200:, :, :]
label_new = sitk.GetImageFromArray(label_)
sitk.WriteImage(label_new, '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/label/pat113_new.nii.gz')
