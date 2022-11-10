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
import scipy
from scipy import stats

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

# def get_max_permutation(scores):
#     # scores: B,5,5
#     # 0 1 2 3 4
#     all_perms = torch.tensor(list(permutations(range(5)))).to(scores.device) # [120, 5]
#     perms_scores = torch.zeros((scores.shape[0], 120)).to(scores.device) # b,120
#     for b in range(all_perms.shape[1]-1):        
#         perms_scores[:] += scores[:, all_perms[:, b], all_perms[:, b+1]]
#     # print(f"all_perms: {all_perms}")
#     # print(f"perms_scores: {perms_scores}")
#     # torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5
#     print(f"all_perms[torch.argmax(perms_scores, dim=1)]: {all_perms[torch.argmax(perms_scores, dim=1)]}")
#     print(f"torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]): {torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1])}")
#     return torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5

# scores = torch.randn(2,5,5)
# # print(scores)
# get_max_permutation(scores)

# def spearman_acc(story, gt_order):
#     return scipy.stats.spearmanr(story, gt_order)[0]

# def absolute_distance(story, gt_order):
#     return np.mean(np.abs(np.array(story) - np.array(gt_order)))

def pairwise_acc(story, gt_order):
    correct = 0
    try:
        story = story.cpu().numpy().tolist()
    except:
        story = story.tolist()
    try:
        gt_order = gt_order.cpu().numpy().tolist()
    except:
        gt_order = gt_order.tolist()
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            gt_order.index(story[idx1])
            if gt_order.index(story[idx1]) < gt_order.index(story[idx2]):
                correct += 1
    return correct/total

# # all_perms = list(permutations(range(5)))
# all_perms = np.array([2,3,1,4,0])
# gt_order = np.array([3,4,2,1,0])
# print(f"spearman_acc: {spearman_acc(all_perms, gt_order)}")
# print(f"absolute_distance: {absolute_distance(all_perms, gt_order)}")
# print(f"pairwise_acc: {pairwise_acc(all_perms, gt_order)}")

all_perms = [2, 3, 1, 4, 0] #  -> [4, 2, 0, 1, 3]
gt_order = [3, 4, 2, 1, 0] # -> [4, 3, 2, 0, 1]

# gt order 原本比如是 3，2，4，0，1 -> 0,1,2,3,4
# predict的story是 4，0，2，3，1 -> 
new_perms = [all_perms.index(i) for i in range(5)]
gt_rank = [gt_order.index(i) for i in range(5)]
# for i in range(5):
#     gt_rank_i = gt_order.index(i)
#     pred_rank_i = all_perms.index(i)
print(new_perms)
print(gt_rank)
