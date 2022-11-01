import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

# from group_modules import *
# from cbam import CBAM
from torchsummary import summary
from einops import rearrange

# Segmentation
class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 2048
        
    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        x = self.avgpool(f16) # 1024
        
        return x.flatten(start_dim=1)

# Clip 不用管了

# ImageNet
class ImageNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024
        self.layer4 = network.layer4 # 1/32, 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 2048
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024
        x = self.layer4(x) # 1/32, 2048
        x = self.avgpool(x) # 2048
        
        return x.flatten(start_dim=1)


if __name__ == '__main__':
    # model = FrameEncoder()
    clip_model, _ = clip.load("RN50")
    img = clip_model.encode_image(torch.randn(1, 3, 224, 224).cuda())
    print(img.shape)
    # model = ImageNetEncoder()
    # print(model)