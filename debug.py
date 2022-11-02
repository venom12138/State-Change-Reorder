import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

from torchsummary import summary
from einops import rearrange
from dataset.EPIC_testdataset import EPICtestDataset

val_dataset = EPICtestDataset(data_root='./val_data', yaml_root='./val_data/EPIC100_state_positive_val.yaml', 
                            valset_yaml_root='./val_data/reordering_val.yaml', num_frames=5, repr_type='ImageNet')

data = val_dataset[3]
print(data['gt_order'])