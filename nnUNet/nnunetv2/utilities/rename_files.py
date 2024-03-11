import os
import shutil
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import random
random.seed(42)
# path1 = '/home/turenzhe/Test/0'
# path2 = '/home/turenzhe/Test/1'
img_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/new'
img1_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTs1'
# lab_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTs'
# temp_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/temp_dir'
for file in os.listdir(img1_dir):
    print(file)
    new_filename = file.split('.')[0]+'.nii.gz'
    nrrd_image = os.path.join(img1_dir, file)
    nrrd_image = sitk.ReadImage(nrrd_image)
    sitk.WriteImage(nrrd_image, os.path.join(img_dir, new_filename))

# for file in os.listdir(path2):
#     if file.endswith('.gz'):
#         filename = os.path.join(path2, file)
#         # print(filename)
#         file_ = file.split('/')[-1].split('.')[0]+'.nii.gz'
#         # print(file_)
#         new_file_name = os.path.join(path2, file_)
#         shutil.move(filename, new_file_name)

# for file in os.listdir(path2):
#     filename = os.path.join(path2, file)
#     print(filename)
#     f = sitk.ReadImage(filename)
#     f = sitk.GetArrayFromImage(f)
#     f[f==3] = 4
#     f[f==5] = 3
#     ff = sitk.GetImageFromArray(f)
#     sitk.WriteImage(ff, filename)

# dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/imagesTs20'
# dir1 = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/images40'
# dir2 = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/imagesTs'
# label_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/labelsTr'
# labelTs_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/labelsTs20'
# src_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTs'

# for file in subfiles(dir1):
#     f = sitk.ReadImage(file)
#     spa = f.GetSpacing()
#     f = sitk.GetArrayFromImage(f)
#     print(spa)

# for file in subfiles(labelTs_dir):
#     # print(file)
#     num = int(file.split('/')[-1][3:6])
#     new_name = str(num+41).zfill(3)
#     print(new_name)
#     new_file = os.path.join(target_dir, 'pat'+new_name+'_0000.nii.gz')
#     new_file = os.path.join(target_dir, 'pat'+new_name+'.nii.gz')
#     shutil.copy(file, new_file)

# #random select test
# test_nums = sorted(random.sample(range(1,61),20))
# print(test_nums)
# trg_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset402_SSLVessel/imagesTs'
# for num in test_nums:
#     name = 'pat'+str(num).zfill(3)+'_0000.nii.gz'
#     print(name)
#     label_name = 'pat'+str(num).zfill(3)+'.nii.gz'
#     print(label_name)
#     # pass
#     src_file = os.path.join(img_dir, name)
#     trg_file = os.path.join(trg_img_dir, name)
#     print(src_file)
#     print(trg_file)
#     shutil.move(src_file, trg_file)
#     src_label_file = os.path.join(lbl_dir, label_name)
#     print(src_label_file)
#     trg_label_file = os.path.join(trg_label_dir, label_name)
#     print(trg_label_file)
#     shutil.move(src_label_file, trg_label_file)

# for file in subfiles(dir):
#     a = sitk.ReadImage(file)
#     a = sitk.GetArrayFromImage(a)
#     print(a.shape) #(D,512,512)
# for file in subfiles(label_dir):
#     file_name = file.split('/')[-1]
#     # print(file_name)
#     # exit()
#     a = sitk.ReadImage(file)
#     a = sitk.GetArrayFromImage(a)
#     b = a.transpose(2,1,0)
#     b= sitk.GetImageFromArray(b)
#     sitk.WriteImage(b, os.path.join('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_ccta_vessel/labels40',file_name))
    # print(a.shape) #(512,512,D)
# for file in subfiles(dir2):
#     a = sitk.ReadImage(file)
#     a = sitk.GetArrayFromImage(a)
#     print(a.shape) #(D,512,512)
# for file in subfiles(dir_):
#     a = sitk.ReadImage(file)
#     a = sitk.GetArrayFromImage(a)
#     print(a.shape) #(D,512,512)
# adjust to (D, 512, 512)

# a = np.load('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset401_ccta_vessel/nnUNetPlans_3d_fullres/pat001_seg.npy')
# print(a.shape)