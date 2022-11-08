import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

from torchsummary import summary
from einops import rearrange
from dataset.EPIC_testdataset import EPICtestDataset
from itertools import permutations
import numpy as np
# # val_dataset = EPICtestDataset(data_root='./val_data', yaml_root='./val_data/EPIC100_state_positive_val.yaml', 
# #                             valset_yaml_root='./val_data/reordering_val.yaml', num_frames=5, repr_type='ImageNet')

# # data = val_dataset[3]
# # print(data['gt_order'])
# all_perms = np.array(list(permutations(range(3)))) # [120, 5]
# print(all_perms)
# scores = np.random.randn(2,3,3) # [B, 5, 5]
# print(scores)
# perms_scores = np.zeros((2, 6))
# for b in range(all_perms.shape[1]-1):
#     # print('ddddddddd')
#     # print(scores[:, all_perms[:, b], all_perms[:, b+1]])
    
#     perms_scores[:] += scores[:, all_perms[:, b], all_perms[:, b+1]]
    
# print(np.argmax(perms_scores, axis=1))
# print(all_perms[np.argmax(perms_scores, axis=1)])
# # print(perms_scores)
# # print(all_perms)

all_perms = np.array(list(permutations(range(5)))) # [120, 5]
target_perm = np.array([0,1,2,3,4])
print(np.sum(np.abs(all_perms-target_perm))/120)