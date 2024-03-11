import os
import SimpleITK as sitk
import numpy as np
dataset_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/imagesTr'
spacings = []
for f in os.listdir(dataset_path):
    file = os.path.join(dataset_path, f)
    # print(file)
    fi = sitk.ReadImage(file)
    print(fi.GetSpacing())
    spacings.append(fi.GetSpacing())
    # print(np.unique(spacings))
print(spacings)