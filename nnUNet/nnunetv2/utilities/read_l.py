# import nrrd # pip install pynrrd
# import SimpleITK as sitk # pip install nibabel
# import numpy as np
# import os
#
# path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTs'
# new_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/imagesTs1'
# # load nrrd
# for file in os.listdir(path):
#     new_filename = file[:-5]+'.nii.gz'
#     print(new_filename)
#     _nrrd = nrrd.read(os.path.join(path, file))
#     data = _nrrd[0]
#     header = _nrrd[1]
#     print(data.shape)
#     # print(type(data))
#     # print(header)
#     data_itk = sitk.GetImageFromArray(data)
#     sitk.WriteImage(data_itk, os.path.join(new_path, new_filename))

import os
from glob import glob
import numpy as np
import vtk


def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


if __name__ == '__main__':
    baseDir = os.path.normpath(r'/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset400_ccta_plaque/new')
    files = glob(baseDir+'/*.nrrd')
    for file in files:
        m, info = readnrrd(file)
        writenifti(m,  file.replace('.nrrd', '._0000.nii.gz'), info)
