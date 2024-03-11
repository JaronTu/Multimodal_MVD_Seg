import pickle

from batchgenerators.utilities.file_and_folder_operations import load_pickle, write_pickle, save_pickle
import numpy as np
import torch
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component

from typing import Union, Tuple, List, Callable
def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                       labels_or_regions: Union[int, Tuple[int, ...],
                                                                                List[Union[int, Tuple[int, ...]]]],
                                                       background_label: int = 0) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret

# pp_fns = []
# pp_fn_kwargs = []
# kwargs = {'labels_or_regions': 1}
# pp_fn = remove_all_but_largest_component_from_segmentation
# pp_fns.append(pp_fn)
# pp_fn_kwargs.append(kwargs)
# # pat[0] = 'remove_all_but_largest_component_from_segmentation'
# # pat[1] = '1'
# # write_pickle(pat,'/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_results/Dataset403_CASVessel/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/postprocessing.pkl')
# save_pickle((pp_fns, pp_fn_kwargs), '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_results/Dataset403_CASVessel/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/postprocessing.pkl')
a = load_pickle('/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_results/Dataset403_CASVessel/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_4/postprocessing.pkl')
print(a)