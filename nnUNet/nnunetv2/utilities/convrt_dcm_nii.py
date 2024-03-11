import os
import pydicom
import SimpleITK
from batchgenerators.utilities.file_and_folder_operations import *


def dicom_to_nifti(dicom_dir, nifti_file):
    dicom_reader = SimpleITK.ImageSeriesReader()
    dicom_names = dicom_reader.GetGDCMSeriesFileNames(dicom_dir)
    print(dicom_names)
    dicom_reader.SetFileNames(dicom_names)
    dicom_image = dicom_reader.Execute()
    SimpleITK.WriteImage(dicom_image, nifti_file)

# convert image files from dicom to nifti
dicom_dir = '/home/siat/PycharmProjects/vessel_label/images/cases'
nifti_output_dir = '/home/siat/PycharmProjects/vessel_label/images'

for (i,file) in enumerate(subdirs(dicom_dir)):
    print(i)
    print(file)
    # pass
    nifti_name = os.path.join(nifti_output_dir, 'pat'+str(i).zfill(3)+'.nii.gz')
    dicom_d = os.path.join(dicom_dir, file, 'PA0', 'ST0', 'SE0')
    dicom_to_nifti(dicom_d, nifti_name)

## convert label files from dicom to nifti
# dicom_dir = '/home/siat/PycharmProjects/vessel_label/labels'
# nifti_output_dir = '/home/siat/PycharmProjects/vessel_label/labels'
# # print(subdirs(dicom_dir))
# # exit()
#
# for (i,file) in enumerate(subdirs(dicom_dir)):
#     # print(file)
#     name = file.split('/')[-1]
#     nifti_name = os.path.join(nifti_output_dir, name+'.nii.gz')
#     dicom_to_nifti(file, nifti_name)