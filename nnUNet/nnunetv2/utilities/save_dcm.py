import SimpleITK as sitk
import numpy as np

reader = sitk.ImageSeriesReader()
img_names = reader.GetGDCMSeriesFileNames('/home/turenzhe/turzh/Datasets/CCTA/vessel_label/ScalarVolume_32')
reader.SetFileNames(img_names)
image = reader.Execute()
# image_array = sitk.GetArrayFromImage(image) # z, y, x

# print(image_array.shape)
# print(np.unique(image_array))
# imga = sitk.
sitk.WriteImage(image, '/home/turenzhe/turzh/Datasets/CCTA/vessel_label/1.nii.gz')