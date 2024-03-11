import numpy as np
from medpy import metric
import SimpleITK as sitk
import os
from PIL import Image
class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):
        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full

def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


if __name__ == '__main__':
    predict_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/inferTs_airway_f0'
    gt_path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV/labelsTs'
    hd1 = []
    hd2 = []
    hd3 = []
    hd4 = []
    # assd1 = []
    # assd2 = []
    for file in sorted(os.listdir(predict_path)):
        if file.endswith('nii.gz'):
            pred_file = os.path.join(predict_path, file)
            print(pred_file)
            pred = sitk.ReadImage(pred_file)
            pred = sitk.GetArrayFromImage(pred)
            gt_file = os.path.join(gt_path, file)
            print(gt_file)
            gt = sitk.ReadImage(gt_file)
            gt = sitk.GetArrayFromImage(gt)
            # pred_img = np.array(Image.open(pred_file))
            # gt_img = np.array(Image.open(gt_file))
            # print(gt_img.shape)
            # print(np.unique(gt_img[:,:,0]==gt_img[:,:,1]))
            # print(np.unique(gt_img))
            # exit()
            # # print(pred.shape)
            # print(np.unique(pred))
            # # print(gt.shape)
            # print(np.unique(gt))
            # print(pred.shap
            target1_mask = (pred==1).astype(np.uint8)
            target2_mask = (pred==2).astype(np.uint8)
            target3_mask = (pred==3).astype(np.uint8)
            target4_mask = (pred==4).astype(np.uint8)
            label1_mask = (gt==1).astype(np.uint8)
            label2_mask = (gt==2).astype(np.uint8)
            label3_mask = (gt==3).astype(np.uint8)
            label4_mask = (gt==4).astype(np.uint8)
            # # for i in
            metric_hd1 = metric.hd95(target1_mask, label1_mask)
            metric_hd2 = metric.hd95(target2_mask, label2_mask)
            metric_hd3 = metric.hd95(target3_mask, label3_mask)
            metric_hd4 = metric.hd95(target4_mask, label4_mask)
            # print(metric_hd1)
            # print(metric_hd2)
            # print(metric_hd3)
            # print(metric_hd4)
            # metric_assd1 = metric.assd(target1_mask, label1_mask)
            # metric_assd2 = metric.assd(target2_mask, label2_mask)
            hd1.append(metric_hd1)
            hd2.append(metric_hd2)
            hd3.append(metric_hd3)
            hd4.append(metric_hd4)
            # assd1.append(metric_assd1)
            # assd2.append(metric_assd2)

    print(hd1)
    print(hd2)
    print(hd3)
    print(hd4)
    # print(assd1)
    # print(assd2)
    hd_avg1 = np.mean(hd1, axis=0)
    hd_avg2 = np.mean(hd2, axis=0)
    hd_avg3 = np.mean(hd3, axis=0)
    hd_avg4 = np.mean(hd4, axis=0)
    print(hd_avg1)
    print(hd_avg2)
    print(hd_avg3)
    print(hd_avg4)
    print(hd_avg1/8.793726670749134  *0.938)
    print(hd_avg2/9.17169992523701   *7.809)
    print(hd_avg3/20.092389901599965 *9.937)
    print(hd_avg4/3.3810773690942977 *21.30)

    # print(assd_avg1)
    # print(assd_avg2)
    # np.array([1.175376019658673,
    # 8.75321562920403,
    # 7.318427207390472,
    # 21.73927008937151])