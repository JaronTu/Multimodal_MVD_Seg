# check whether plaque inference are in vessel inference
import os
import SimpleITK as sitk
import numpy as np

plaque_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTs'
vessel_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset403_CASVessel/inferTs_plaque_PP'

for file in os.listdir(plaque_dir):
    if not file.endswith('.json'):
        plaque_path = os.path.join(plaque_dir, file)
        vessel_path = os.path.join(vessel_dir, file)
        plaque = sitk.ReadImage(plaque_path)
        plaque = sitk.GetArrayFromImage(plaque)
        vessel = sitk.ReadImage(vessel_path)
        vessel = sitk.GetArrayFromImage(vessel)
        cross_label = np.sum(plaque * vessel)
        # print(cross_label)
        # if cross_label == 0:
        #     print(file)
        shape = plaque.shape
        all_sum = shape[0] * shape[1] * shape[2]
        print(all_sum)
        plaque_sum = np.sum(plaque)
        print(plaque_sum)
        portion = plaque_sum/all_sum
        print(portion)