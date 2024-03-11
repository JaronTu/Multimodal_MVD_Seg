# aggregate plaque label and vessel label
import numpy
import os
import SimpleITK as sitk
import numpy as np
import shutil

dir1 = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset403_CASVessel/inferTs_temp1_pp'
dir2 = '/home/siat/CCTA/masks_segment'
img_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTr'
dir_save = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset403_CASVessel/infer_aggregated'
new_img_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTr_selected'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
for file in os.listdir(dir1):
    print(file)
    file_n = file.split('.')[0] + '_gt.nii.gz'
    filename1 = os.path.join(dir1, file)
    filename2 = os.path.join(dir2, file_n)
    if not os.path.exists(filename2):
        filename2 = os.path.join(dir2, file)
    vessel = sitk.ReadImage(filename1)
    vessel = sitk.GetArrayFromImage(vessel)
    print(vessel.shape)
    plaque = sitk.ReadImage(filename2)
    plaque = sitk.GetArrayFromImage(plaque)
    new_map = np.zeros_like(plaque)
    new_map[vessel == 1] = 1
    new_map[plaque == 1] = 2
    # print(np.unique(new_map))
    new = sitk.GetImageFromArray(new_map)
    sitk.WriteImage(new, os.path.join(dir_save, file))
    img_filename = os.path.join(img_dir, file)
    new_imgname = os.path.join(new_img_dir, file)
    shutil.copy(img_filename, new_imgname)