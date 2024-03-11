import os
import shutil
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import random
random.seed(42)

#random select test
test_nums = sorted(random.sample(range(0,222),40))
print(test_nums)
img_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset405_CroppedRegions/imagesTr'
lab_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset405_CroppedRegions/labelsTr'
trg_img_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset405_CroppedRegions/imagesTs'
trg_label_dir = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset405_CroppedRegions/labelsTs'
for file in os.listdir(img_dir):
    new_file = file[:6]+'_0000.nii.gz'
    # print(new_file)
    # exit()
    file1 = os.path.join(img_dir, file)
    file2 = os.path.join(img_dir, new_file)
    shutil.move(file1, file2 )

for num in test_nums:
    src_img_name = 'pat' + str(num).zfill(3)+'_0000.nii.gz'
    # trg_img_name =
    src_lab_name = 'pat' + str(num).zfill(3)+'.nii.gz'
    shutil.move(os.path.join(img_dir, src_img_name), os.path.join(trg_img_dir, src_img_name))
    shutil.move(os.path.join(lab_dir, src_lab_name), os.path.join(trg_label_dir, src_lab_name))