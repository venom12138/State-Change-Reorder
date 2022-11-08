import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

# from group_modules import *
# from cbam import CBAM
from torchsummary import summary
from einops import rearrange
from decord import VideoReader, cpu
import numpy as np

from transformers import VideoMAEFeatureExtractor, VideoMAEModel
from huggingface_hub import hf_hub_download

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

# Segmentation
class KeyEncoder(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
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

class VideoMae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
        self.model.embeddings.position_embeddings = get_sinusoid_encoding_table(14*14*5, self.model.config.hidden_size)
        if not config['use_position_embedding']:
            print('not use position embedding')
            self.model.embeddings.position_embeddings = torch.zeros_like(self.model.embeddings.position_embeddings, requires_grad=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 768
    
    # B, numframes*2, 3, 224, 224
    def forward(self, x):
        B = x.shape[0]
        num_frames = x.shape[1]//2
        H = x.shape[-2]//16
        W = x.shape[-1]//16
        
        outputs = self.model(x)
        x = outputs.last_hidden_state # B, numframes*14*14, 768
        
        x = x.reshape(B, num_frames, H, W, x.shape[-1])
        x = x.permute(0,1,4,2,3)
        x = self.avgpool(x) # B, numframes, 768
        
        return x.flatten(start_dim=2) # B, numframes, 768

class Classifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_channels*2, in_channels*2),
                                    nn.ReLU(),
                                    nn.Linear(in_channels*2, 2))
    
    def forward(self, x):
        
        return self.classifier(x)

if __name__ == '__main__':
    # model = FrameEncoder()
    clip_model, _ = clip.load("RN50")
    img = clip_model.encode_image(torch.randn(1, 3, 224, 224).cuda())
    print(img.shape)
    # model = ImageNetEncoder()
    # print(model)