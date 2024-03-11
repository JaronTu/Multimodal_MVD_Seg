import matplotlib

matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import os
# import visdom
import skimage.measure
import random
import numpy as np
from tqdm import tqdm as tqdm
import sys
import SimpleITK as sitk
from nnunetv2.training.metrics.betti_compute import betti_number


def getBetti(binaryPredict, masks):
    predict_betti_number_ls = []
    groundtruth_betti_number_ls =[]
    betti_error_ls = []
    topo_size = 65
    gt_dmap = masks.cuda()
    # et_dmap = likelihoodMap_final
    # n_fix = 0
    # n_remove = 0
    # topo_cp_weight_map = np.zeros(et_dmap.shape)
    # topo_cp_ref_map = np.zeros(et_dmap.shape)
    # allWindows = 1
    # inWindows = 1

    for y in range(0, gt_dmap.shape[0], topo_size):
        for x in range(0, gt_dmap.shape[1], topo_size):
            # likelihoodAll = []
            # allWindows = allWindows + 1
            # likelihood = et_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
            #              x:min(x + topo_size, gt_dmap.shape[1])]
            binary = binaryPredict[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]           
            groundtruth = gt_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                          x:min(x + topo_size, gt_dmap.shape[1])]
            # for likelihoodMap in likelihoodMaps:
            #     likelihoodAll.append(likelihoodMap[y:min(y + topo_size, gt_dmap.shape[0]),
            #              x:min(x + topo_size, gt_dmap.shape[1])])

            # print('likelihood', likelihood.shape, 'groundtruth', groundtruth.shape, 'binaryPredict', binary.shape)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)
            # print(predict_betti_number, groundtruth_betti_number)
            predict_betti_number_ls.append(predict_betti_number)
            groundtruth_betti_number_ls.append(groundtruth_betti_number)
            betti_error_ls.append(abs(predict_betti_number-groundtruth_betti_number))

    return betti_error_ls


if __name__ == '__main__':
    predict_path = '/home/turenzhe/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/ERNetTrainer_f3'
    gt_path = '/home/turenzhe/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/labelsTs'
    cldice = []
    for file in sorted(os.listdir(predict_path)):
        if file.endswith('nii.gz'):
            pred_file = os.path.join(predict_path, file)
            gt_file = os.path.join(gt_path, file)
            pred = sitk.ReadImage(pred_file)
            pred = sitk.GetArrayFromImage(pred)
            gt = sitk.ReadImage(gt_file)
            gt = sitk.GetArrayFromImage(gt)
            # print(pred.shape)
            # print(gt.shape)
            # print(np.unique(pred))
            # print(np.unique(gt))
            # print(file)
            cld = []
            for num in range(4):
                pred_ = np.where(pred==num, 1, 0)
                gt_ = np.where(gt == num, 1, 0)
                pred_ = torch.from_numpy(pred_)
                gt_ = torch.from_numpy(gt_)
                metric = getBetti(pred_, gt_)
                cld.append(metric)
            print(cld)
            cldice.append(cld)
    print(cldice)
    cldice_average = np.mean(cldice, axis=0)
    print(cldice_average)