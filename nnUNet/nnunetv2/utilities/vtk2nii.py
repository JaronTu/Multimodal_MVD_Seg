import os.path
from pyvista import _vtk, PolyData
import vtk
from vtkmodules.util import numpy_support
import glob
import SimpleITK as sitk
import numpy as np
from xpinyin import Pinyin

# if __name__ == '__main__':
def read_mesh_file(filename):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    # print(reader)
    polydata = reader.GetOutput()
    return polydata

def polydata_to_imagedata(polydata, dimensions=(100, 100, 100), padding=1):
    xi, xf, yi, yf, zi, zf = polydata.GetBounds()
    dx, dy, dz = dimensions

    # Calculating spacing
    sx = (xf - xi) / dx
    sy = (yf - yi) / dy
    sz = (zf - zi) / dz

    # Calculating Origin
    ox = xi + sx / 2.0
    oy = yi + sy / 2.0
    oz = zi + sz / 2.0

    if padding:
        ox -= sx
        oy -= sy
        oz -= sz

        dx += 2 * padding
        dy += 2 * padding
        dz += 2 * padding

    image = vtk.vtkImageData()
    image.SetSpacing((sx, sy, sz))
    image.SetDimensions((dx, dy, dz))
    image.SetExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    image.SetOrigin((ox, oy, oz))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    inval = 1
    outval = 0

    for i in range(image.GetNumberOfPoints()):
        image.GetPointData().GetScalars().SetTuple1(i, inval)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin((ox, oy, oz))
    pol2stenc.SetOutputSpacing((sx, sy, sz))
    pol2stenc.SetOutputWholeExtent(image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()


def save(imagedata, filename):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imagedata)
    writer.Write()


def main():
    cases_folder = '/home/turenzhe/turzh/Datasets/CCTA/cases'
    folder = '/home/turenzhe/turzh/Datasets/CCTA/cases/安天恩/PA0/ST0/SE0'
    series_reader = sitk.ImageSeriesReader()
    fileNames = series_reader.GetGDCMSeriesFileNames(folder)
    series_reader.SetFileNames(fileNames)
    images = series_reader.Execute()
    data_np = sitk.GetArrayFromImage(images)
    # print(data_np.shape)

    name = folder.split('/')[-4]
    p =Pinyin()
    py_name = p.get_pinyin(name,'')
    # print(py_name)

    input_folder = '/home/turenzhe/turzh/Datasets/CCTA/model'
    input_filename = os.path.join(input_folder, py_name+'.stl')
    # input_folder +
    #     "/home/turenzhe/turzh/Datasets/CCTA/model/antianen.stl"
    print(input_filename)
    dimensions = data_np.shape
    # output_filename = sys.argv[2]
    output_filename = '/home/turenzhe/turzh/Datasets/CCTA/vessel_label'

    polydata = read_mesh_file(input_filename)
    imagedata = polydata_to_imagedata(polydata, dimensions)
    cols, rows, levels = imagedata.GetDimensions()
    sc = imagedata.GetPointData().GetScalars()
    imageArr = numpy_support.vtk_to_numpy(sc)
    image3D = imageArr.reshape(cols, rows, levels)
    # save(image3D, output_filename)
    # print(image3D.shape)
    # print(np.unique(imageArr))
    itk_img = sitk.GetImageFromArray(image3D)
    # save_path =
    sitk.WriteImage(itk_img, os.path.join(output_filename, py_name+'.nii.gz'))


if __name__ == "__main__":
    main()
    # a = '/home/turenzhe/turzh/Datasets/CCTA/masks/pat001_gt.nii.gz'
    # img = sitk.ReadImage(a)
    # img = sitk.GetArrayFromImage(img) (224, 512, 512)
    # print(img.shape)