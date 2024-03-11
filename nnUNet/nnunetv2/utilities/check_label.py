import os
import cv2
import SimpleITK as sitk
import numpy as np

infer_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/inferTs_f4'
label_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTs'

# for file in os.listdir(infer_path):
#     # print(file)
#     if not file.endswith('.json'):
#         infer_f = os.path.join(infer_path, file)
#         label_f = os.path.join(label_path, file)
#         infer = sitk.ReadImage(infer_f)
#         infer = sitk.GetArrayFromImage(infer)
#         # print(label_f)
#         label = sitk.ReadImage(label_f)
#         label = sitk.GetArrayFromImage(label)
#         if infer.shape != label.shape:
#             print(infer.shape)
#             print(label.shape)
#             print(file)
