import torch
import torch.nn as nn
from copy import deepcopy
from model.modules import *
# from aggregate import aggregate
# from modules import *
# from memory_util import *
from torchsummary import summary
import clip

class FrameReorderNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['repr_type'] == 'Clip':
            clip_model, _ = clip.load("RN50") # clip.load("ViT-L/14@336px")
            self.Encoder = clip_model.visual # 输出是2048维的
            self.Classifier = nn.Linear(2048*2, 2)
        
        elif self.config['repr_type'] == 'ImageNet':
            self.Encoder = ImageNetEncoder() # 输出是2048维 
            self.Classifier = nn.Linear(2048*2, 2)
            
        elif self.config['repr_type'] == 'Segmentation':
            self.Encoder = KeyEncoder() # 输出是1024维
            self.Classifier = nn.Linear(1024*2, 2)
        
        elif self.config['repr_type'] == 'Action':
            raise NotImplementedError
        
        else:
            raise NotImplementedError
        
        if config['load_network'] != '':
            model_path = config['load_network']
            print('Loading model from {}'.format(model_path))
            model_weights = torch.load(model_path)
            self.load_model(model_weights)

        if self.config['freeze']:
            print('Freezing the encoder')
            for param in self.Encoder.parameters():
                param.requires_grad = False
                
    def encode(self, x):
        if len(x.shape) == 5:
            batch_size, num_frames = x.shape[:2]
            x = x.flatten(start_dim=0, end_dim=1) 
            need_reshape = True
        else:
            need_reshape = False
        
        x = self.Encoder(x) 
        
        if need_reshape:
            x = x.view(batch_size, num_frames, *x.shape[1:])
            
        return x  # [B, numframes, 2048]
    
    # x: [B,1024*2/2048*2] 
    def classify(self, x):
        return self.Classifier(x) # B,2
    
    # def forward(self, x1, x2):
    #     x1 = self.Encoder(x1) # [B, 2048]
    #     x2 = self.Encoder(x2) # [B, 2048]
    #     x = torch.cat([x1, x2], dim=1) # [B, 2048*2]
    #     x = self.CLassifier(x)
        
    #     return x
    
    def load_model(self, src_dict):
        assert self.config['repr_type'] in ['Segmentation', 'Action']
        if self.config['repr_type'] == 'Segmentation':
            new_src_dict = {}
            for key, value in src_dict.items():
                if key.startswith('key_encoder'):
                    new_key = key.replace('key_encoder.', '')
                    new_src_dict.update({new_key: value})
            self.Encoder.load_state_dict(new_src_dict)
        
        elif self.config['repr_type'] == 'Action':
            raise NotImplementedError