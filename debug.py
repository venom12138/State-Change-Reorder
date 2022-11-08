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
def get_max_permutation(scores):
    # scores: B,5,5
    # 0 1 2 3 4
    all_perms = torch.tensor(list(permutations(range(5)))).to(scores.device) # [120, 5]
    perms_scores = torch.zeros((scores.shape[0], 120)).to(scores.device) # b,120
    for b in range(all_perms.shape[1]-1):        
        perms_scores[:] += scores[:, all_perms[:, b], all_perms[:, b+1]]
    # print(f"all_perms: {all_perms}")
    # print(f"perms_scores: {perms_scores}")
    # torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5
    print(f"all_perms[torch.argmax(perms_scores, dim=1)]: {all_perms[torch.argmax(perms_scores, dim=1)]}")
    print(f"torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]): {torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1])}")
    return torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5

scores = torch.randn(2,5,5)
# print(scores)
get_max_permutation(scores)