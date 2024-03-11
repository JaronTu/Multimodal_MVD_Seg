import SimpleITK as sitk
import os
import numpy as np
path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/inferTs_f0'
new_save = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/inferTs_f0_plaque'
for file in os.listdir(path):
    if file.endswith('.gz'):
        filename = os.path.join(path, file)
        # print(filename)
        a = sitk.ReadImage(filename)
        a = sitk.GetArrayFromImage(a)
        new_a = np.zeros_like(a)
        new_a[a==2]=1
        b = sitk.GetImageFromArray(new_a)
        print(os.path.join(new_save, file))
        sitk.WriteImage(b, os.path.join(new_save, file))