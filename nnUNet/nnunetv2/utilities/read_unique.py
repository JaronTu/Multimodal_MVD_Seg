import SimpleITK as sitk
import numpy as np
import os

path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTr/pat001_0000.nii.gz'
# for file in sorted(os.listdir(path)):
#     filename = os.path.join(path, file)
#     print(filename)
#     if filename.endswith('.nii.gz'):
#         f = sitk.ReadImage(filename)
#         f = sitk.GetArrayFromImage(f)
#         print(np.unique(f))
#         print(file)
labPath = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTs'
for file in sorted(os.listdir(labPath)):
    filename = os.path.join(labPath, file)
    print(filename)
    if filename.endswith('.nii.gz'):
        f = sitk.ReadImage(filename)
        f = sitk.GetArrayFromImage(f)
        print(np.unique(f))
# print(f.shape)