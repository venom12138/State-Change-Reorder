import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

# from group_modules import *

# from cbam import CBAM
from torchsummary import summary
from einops import rearrange


if __name__ == '__main__':
    model_weights = torch.load('/home/venom/.exp/1027_more_numframes/D0169_freeze=0,fuse_type=cross_attention,num_frames=20,steps=1000,use_text=0,use_flow=1/network_6000.pth', map_location=None)
    for key,_ in model_weights.items():
        print(key)