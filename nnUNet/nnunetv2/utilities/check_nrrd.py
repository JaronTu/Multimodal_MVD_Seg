import nrrd
import SimpleITK as sitk
import shutil
from batchgenerators.utilities.file_and_folder_operations import *

ccta_base_dir = '/home/siat/CCTA/ASOCAData/Normal/CTCA'
cases = subfiles(ccta_base_dir)
# label_cases = subfiles(ccta_label_dir)

for i, tr in sorted(enumerate(cases)):
    path = join(ccta_base_dir, tr)
    print(path)
    data, options = nrrd.read(path)
    print(options)
    # itk_data = sitk.GetImageFromArray(data)
    # trl = 'pat'+str(i+1).zfill(3)
    # sitk.WriteImage(itk_data, join(imagestr, f'{trl}_0000.nii.gz'))
    # print(tr)
    # path = join(ccta_label_dir, tr)
    # print(path)
    # data, options = nrrd.read(path)