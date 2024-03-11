from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import SimpleITK as sitk
import os
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def cal_clDice(predict_path, gt_path='/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/labelsTs'):
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
            cl_dc = clDice(pred, gt)
            # print(file)
            cld = []
            for num in range(4):
                pred_ = np.where(pred == num, 1, 0)
                gt_ = np.where(gt == num, 1, 0)
                metric = clDice(pred_, gt_)
                cld.append(metric)
            # print(cld)
            cldice.append(cld)
    # print(cldice)
    cldice_average = np.mean(cldice, axis=0)
    # print(cldice_average)
    return cldice_average

if __name__ == '__main__':
    predict_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/inferTs_cldice_f0'
    gt_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/labelsTs'
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
            cl_dc = clDice(pred, gt)
            # print(file)
            cld = []
            for num in range(4):
                pred_ = np.where(pred==num, 1, 0)
                gt_ = np.where(gt == num, 1, 0)
                metric = clDice(pred_, gt_)
                cld.append(metric)
            print(cld)
            cldice.append(cld)
    print(cldice)
    cldice_average = np.mean(cldice, axis=0)
    print(cldice_average)
    # a = [[0.9783060252284735, 0.7697841726618705, 0.6826336057105288, 0.5462784257783], [0.9778646865533374, 0.8400529649452166, 0.6723280977965816, 0.461459474260679], [0.9855289549789535, 0.9304029304029303, 0.6663031357002146, 0.6534629277400666], [0.9867829454925179, 1.0, 0.6313101593039384, 0.7481415865643593], [0.980286526235746, 0.8365758754863813, 0.7396224899137431, 0.5903678569207274], [0.971345722414234, 0.6544622425629291, 0.677355118713301, 0.4171146623462699], [0.9799252909783794, 0.8962207558488302, 0.7693078564482316, 0.6994231323911162], [0.9907589242274185, 0.9452332657200813, 0.7971016486350787, 0.6711462717218997], [0.9843436927531694, 0.9635719134173015, 0.7710094242313119, 0.4809525820208589], [0.9896286039735169, 0.9597420271967909, 0.8702126485600565, 0.6292727566345481], [0.9948471914090268, 0.985781990521327, 0.6433246174554794, 0.7023550640818027], [0.9829189843592285, 0.9375, 0.7496450103587142, 0.6456010516578218], [0.9772664842615957, 0.853229220784256, 0.6557298393248311, 0.5491914372180242], [0.9867378430315026, 0.9274056568938778, 0.7603838669845595, 0.666427803654604], [0.991537686546703, 0.9356429384245797, 0.7726462861311949, 0.6019890260631001], [0.9880504090747242, 0.9189189189189189, 0.799455094560118, 0.6152559225997885]]
    # avg = np.mean(a, axis=0)
    # print(avg)