import os
import shutil

base_path = '/home/turenzhe/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV'
source_label_path = '/home/turenzhe/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset334_nervessel/labelsTr'
target_label_path = '/home/turenzhe/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/labelsTr'
image_path = os.path.join(base_path, 'imagesTr')
label_name_list = []

for path in sorted(os.listdir(image_path)):
    label_path = os.path.join(base_path, 'labelsTr')
    # print(path)
    num = path[3:6]
    # print(num)
    if num != '095':
        label_name = 'pat'+num+'.nii.gz'
        label_name_list.append(label_name)
        shutil.copy(os.path.join(source_label_path, label_name), os.path.join(target_label_path, label_name))