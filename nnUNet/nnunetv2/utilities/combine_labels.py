import torch
import numpy as np
import os
import SimpleITK as sitk
# name = ['8', '40', '41']
name = ['120', '121']

root_path = '/home/siat/CCTA/images'
mask_path = '/home/siat/CCTA/masks'
save_path = '/home/siat/CCTA/masks_segment'
d={}
i=0
for n in name:
    i+=1
    n_name = 'pat'+str(n).zfill(3)+'.nii.gz'
    path = os.path.join(root_path, n_name)
    label_name ='pat'+str(n).zfill(3)+'_gt.nii.gz'
    label_path = os.path.join(mask_path, label_name)
    if not os.path.exists(label_path):
        label_path = os.path.join(mask_path, 'pat'+str(n).zfill(3)+'.nii.gz')
    print(label_path)
    lab = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(lab)
    # a = np.zeros_like(label)
    # b = np.logical_or(a, b)
    # print(np.unique(label))
    var_name = "a"+str(i)
    d[var_name] = label
final_label = np.logical_or(d['a1'], d['a2']).astype(int)
# print(np.unique(final_label))
# final_label = np.logical_or(final_label, d['a3']).astype(int)
save_name = os.path.join(save_path, 'pat'+str(120).zfill(3)+'_gt.nii.gz')
ssvv = sitk.GetImageFromArray(final_label)
sitk.WriteImage(ssvv, save_name)
# print(len(d))
# exit()
# a_1 = sitk.ReadImage(path1)
# a_2 = sitk.ReadImage(path2)
# a_3 = sitk.ReadImage(path3)
# a_1 = sitk.GetArrayFromImage(a_1)
# a_2 = sitk.GetArrayFromImage(a_2)
# a_3 = sitk.GetArrayFromImage(a_3)
# A =