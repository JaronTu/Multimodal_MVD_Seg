import os
import shutil
import numpy as np

# path = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset334_nervessel/imagesTr'
# target_path = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_TOF/imagesTr'
# for file in os.listdir(path):
#     filename = os.path.join(path, file)
#     target_filename = os.path.join(target_path, file)
#     if filename.split('.')[0].endswith("1"):
#         shutil.copy(filename, target_filename)

target_path = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_TOF/imagesTs'

# 获取文件夹中的所有文件名
file_list = os.listdir(target_path)

# 遍历文件列表，将文件名更改为新的文件名
for filename in file_list:
    new_f = filename[:-7]
    new_f = new_f[:-1] + "0"
    new_filename = new_f + ".nii.gz"
    print(new_filename)
    new_filepath = os.path.join(target_path, new_filename)
    old_filepath = os.path.join(target_path, filename)
    os.rename(old_filepath, new_filepath)