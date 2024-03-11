import os
import shutil

path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_plaque_single/labelsTr'
none_files = []
path2 = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTr'
test_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset401_plaque_single/labelsTr'
target_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/labelsTs'
two_files = []
for file in os.listdir(test_path):
        print(file)
        target_name = os.path.join(target_path, file)
        gt_name = file[:6]+'_gt.nii.gz'
        gt_file = os.path.join(test_path, gt_name)
        if not os.path.exists(gt_file):
                gt_file = os.path.join(test_path, file)
        print(gt_file)
        # exit()
        shutil.move(gt_file, target_name)