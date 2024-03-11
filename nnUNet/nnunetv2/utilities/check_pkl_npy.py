import numpy as np
import pickle as pkl

npy_path = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset333_nervessel/nnUNetPlans_3d_fullres/pat001_seg.npy'
npfile = np.load(npy_path)
# print(npfile)
# print(npfile.shape)
# print(np.unique(npfile))
pkl_path = '/home/hci/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_preprocessed/Dataset333_nervessel/nnUNetPlans_3d_fullres/pat001.pkl'
with open(pkl_path,'rb') as f:
    pkl_file = pkl.load(f)
print(pkl_file['sitk_stuff'])