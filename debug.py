import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

from torchsummary import summary
from einops import rearrange
from dataset.EPIC_testdataset import EPICtestDataset, EPICCliptestDataset
from dataset.EPIC_dataset import EPICClipDataset
from itertools import permutations
import numpy as np
import scipy
from scipy import stats

# val_dataset = EPICClipDataset(data_root='./EPIC_train', yaml_root='./EPIC_train/EPIC100_state_positive_train.yaml', 
#                             num_frames=5, repr_type='ImageNet') # valset_yaml_root='./val_data/reordering_val.yaml', 
# data = val_dataset[2]
# print(f"images: {data['rgb'].shape}")
# val_dataset_2 = EPICCliptestDataset(data_root='./val_data', yaml_root='./val_data/EPIC100_state_positive_val.yaml', 
#                             valset_yaml_root='./val_data/reordering_val.yaml', num_frames=5, repr_type='ImageNet') # 
# data = val_dataset_2[2]
# print(f"images: {data['rgb'].shape}")
input_feat = torch.randn(2, 5, 8, 256, 16, 16)
pooling_feat = torch.mean(input_feat, dim=2)

print(pooling_feat.shape)