import os
import numpy as np
from PIL import Image
import SimpleITK as sitk

# path = '/home/siat/turzh/FedICRA-master/data/ODOC/Domain1/train/imgs'
# for file in os.listdir(path):
#     filename = os.path.join(path, file)
#     rf = Image.open(filename).convert("RGB")
#     # pixels = rf.load()
#     rf_np = np.array(rf)
#     print(np.unique(rf_np[0]==rf_np[1]))

path = '/home/siat/Downloads/LNDb-0001_rad1.mhd'
file = sitk.ReadImage(path)
file = sitk.GetArrayFromImage(file)
print(file.shape)
print(np.unique(file))