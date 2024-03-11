import os
import shutil

image_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_vessel_plaque/imagesTr'
lab_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_vessel_plaque/labelsTr'

for file_name in os.listdir(image_path):
    # print(file_name)
    new_name = file_name[:6]+'_0000.nii.gz'
    lab_name = os.path.join(lab_path, file_name)
    if os.path.exists(lab_name):
        shutil.move(os.path.join(image_path, file_name), os.path.join(image_path, new_name))
    else:
        os.remove(os.path.join(image_path, file_name))