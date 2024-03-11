import os
import shutil
from skimage import io
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset101_Eye1/labelsTs'
post_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset101_Eye1/labelTs_postpro'
for input_seg in subfiles(path):
    filename = input_seg.split('/')[-1]
    if filename.endswith('png'):
        seg = io.imread(input_seg)
        seg = np.uint8(seg * 85)
        output_seg = os.path.join(post_path, filename)
        io.imsave(output_seg, seg, check_contrast=False)
    # shutil.copy(input_image, output_image)