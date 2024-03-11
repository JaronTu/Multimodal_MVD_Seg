import numpy as np
import scipy.stats as stats
import os
import json
from nnunetv2.training.metrics.clDice_metric import cal_clDice
# files = ['inferTs_cldice_f0', 'inferTs_cldice_f1', 'inferTs_cldice_f2', 'inferTs_cldice_f3', 'inferTs_cldice_f4']
# files = ['inferTs_airway_f0', 'inferTs_airway_f1', 'inferTs_airway_f2', 'inferTs_airway_f3', 'inferTs_airway_f4']
files = ['inferTs_f0', 'inferTs_f1', 'inferTs_f2', 'inferTs_f3', 'inferTs_f4']
# files = ['ERNetTrainer_f3', 'ERNetTrainer_f4']
# files = ['inferTs_unetrppv1_f3', 'inferTs_unetrppv1_f4']
# files = ['inferTs_selfattnFv2_f3', 'inferTs_selfattnFv2_f4']
# files = ['SHAPRTrainer_f3','SHAPRTrainer_f4']
# files = ['HDenseFormerTrainer_f0','HDenseFormerTrainer_f1', 'HDenseFormerTrainer_f3', 'HDenseFormerTrainer_f4']
path = '/home/siat/PycharmProjects/nnUNetFrameV2/DATASET/nnUNet_raw/Dataset336_tempNV'
dice = []
# for file in files:
#     dir = os.path.join(path, file, 'summary.json')
#     dice_fold = []
#     with open(dir, 'r') as f:
#         ff = json.load(f)
#         dice1 = ff['mean']['1']['Dice']
#         dice2 = ff['mean']['2']['Dice']
#         dice3 = ff['mean']['3']['Dice']
#         dice4 = ff['mean']['4']['Dice']
#         dice_fold.append(dice1)
#         dice_fold.append(dice2)
#         dice_fold.append(dice3)
#         dice_fold.append(dice4)
#         dice.append(dice_fold)
# # print(len(dice))
# dice = np.array(dice)*100
# # print(dice)
# data_split = np.split(dice, 4)
# # print(data_split)
# mean = np.mean(data_split,axis=0)
# std = np.std(data_split, axis=0,ddof=1)
# print(mean)
# print(std)

clDice = []
for file in files:
    dir = os.path.join(path, file)
    cldice_fold = []
    cldice = cal_clDice(dir)
    clDice.append(cldice)

# # print(len(dice))
clDice = np.array(clDice)*100
print(clDice)
data_split = np.split(clDice, 5)
# print(data_split)
mean = np.mean(data_split,axis=0)
std = np.std(data_split, axis=0,ddof=1)
print(mean)
print(std)