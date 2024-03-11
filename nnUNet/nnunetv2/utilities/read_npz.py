import numpy as np
import pickle
import SimpleITK as sitk
from PIL import Image
import scipy.misc
import os
import shutil

# stage0 = np.load("/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_preprocessed/Task333_nervessel/nnUNetData_plans_v2.1_stage0/pat001.npz")
# image_array = stage0['data']
# print(stage0['data'].shape)

# croppe_data = np.load('/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_cropped_data/Task335_NVsinglemoda/pat001.npz')
# print(croppe_data['data'].shape)
# exit()
# out = sitk.GetImageFromArray(image_array[0])
# sitk.WriteImage(out, '/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_preprocessed/Task333_nervessel/preprocessed_niigz/001.nii.gz')


# file = open("/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_preprocessed/Task333_nervessel/nnUNetData_plans_v2.1_stage0/pat001.pkl", "rb")
# data = pickle.load(file)
# print(data)
# print(data.keys())
# print(data['list_of_data_files']) #[ 72 960 960]
# print(data['size_after_cropping'])
# print(data['seg_file'])
# print(data['original_spacing'])#[0.50000006 0.19791667 0.19791667]
# print(data['size_after_resampling']) #(72, 380, 359)
# print(data['spacing_after_resampling']) #[0.5 0.5 0.5]
# print(data['class_locations'])
# print(data['class_locations'])
# file = open('/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_preprocessed/Task333_nervessel/nnUNetPlansv2.1_plans_3D.pkl','rb')
# data = pickle.load(file)
# print(data.keys())
# # print(data['plans_per_stage'])
# # print(data['preprocessed_data_folder'])
# print("dataset_properties",data['dataset_properties'])
# print("original_spacings",data['original_spacings'])
# print("original_sizes",data['original_sizes'])
# print("preprocessed_data_folder",data['preprocessed_data_folder'])
# print(data['plans_per_stage'])
# print(data['normalization_schemes'])
# npy_data = np.load('/home/hci/turzh/nnUNetFrame/DATASET/nnUNet_preprocessed/Task333_nervessel/nnUNetData_plans_v2.1_stage0/pat001.npy')
# print(npy_data.shape)
# print(np.unique(npy_data))
npy_data = np.load('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset336_tempNV/nnUNetPlans_3d_fullres/pat019.npy')
print(npy_data.shape)
# npz_data = np.load('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset400_ccta_plaque/nnUNetPlans_3d_fullres/pat001.npz')

# npz_data = np.load('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset403_CASVessel/nnUNetPlans_3d_fullres/pat001.npz')
# print(npz_data['seg'].shape)
# print(npz_data['data'].shape)
# preprocessed_folder = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset400_ccta_plaque/nnUNetPlans_3d_fullres'
# for f in os.listdir(preprocessed_folder):
#     f_path = os.path.join(preprocessed_folder, f)
#     # print(f)
#     if f.endswith('_seg.npy'):
#         npy_data = np.load(f_path)
#         print(npy_data.shape)
#
