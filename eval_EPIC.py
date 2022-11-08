import os
from os import path
from argparse import ArgumentParser
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from dataset.EPIC_testdataset import EPICtestDataset
from dataset.EPIC_dataset import EPICDataset
from model.network import FrameReorderNet
import scipy
from scipy import stats

from progressbar import progressbar
from tqdm import tqdm
import clip
from itertools import permutations
from torch.optim.swa_utils import update_bn

# scores: [B, 5, 5]
def get_max_permutation(scores):
    # scores: B,5,5
    # 0 1 2 3 4
    all_perms = torch.tensor(list(permutations(range(5)))).to(scores.device) # [120, 5]
    perms_scores = torch.zeros((scores.shape[0], 120)).to(scores.device) # b,120
    for b in range(all_perms.shape[1]-1):        
        perms_scores[:] += scores[:, all_perms[:, b], all_perms[:, b+1]]
    # print(f"perms_scores: {perms_scores}")
    return torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5


def spearman_acc(story, gt_order):
    return scipy.stats.spearmanr(story, gt_order)[0]

def absolute_distance(story, gt_order):
    return np.mean(np.abs(np.array(story) - np.array(gt_order)))

def pairwise_acc(story, gt_order):
    correct = 0
    # gt order 原本比如是 3，2，4，0，1
    # predict的story是 4，0，2，3，1
    # 那么将3：0，2：1，4：2，0：3，1：4做这样一个替换
    # story就变成了 2，3，1，0，4
    # 然后gt_order就变为了0，1，2，3，4
    for i in range(len(gt_order)):
        index = story.index(gt_order[i])
        story[index] = i
        
    gt_order = list(range(len(gt_order)))
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            if story[idx1] < story[idx2]:
                correct += 1
    return correct/total

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='/home/venom/.exp/1102_reordering/D0184_freeze=0,repr_type=ImageNet,cos_lr=True,bs=8,iter=50000,lr=1e-3/network_50000.pth')
parser.add_argument('--load_network', default='')

# Data options
parser.add_argument('--EPIC_path', default='./val_data')
parser.add_argument('--yaml_path', default='./val_data/EPIC100_state_positive_val.yaml')
parser.add_argument('--valset_yaml_path', default='./val_data/reordering_val.yaml')
parser.add_argument('--output', default=None)
parser.add_argument('--num_frames', default=5, type=int)
parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
parser.add_argument('--repr_type', type=str, choices=['Clip', 'ImageNet', 'Segmentation', 'Action'])
parser.add_argument('--freeze', default=0, type=int, choices=[0,1])
args = parser.parse_args()

config = vars(args)
assert config['load_network'] == ''

if args.output is None:
    # args.output = f'./output/{args.dataset}_{args.split}'
    args.output = f"./output/{args.load_network.split('/')[-1][:-4]}"
    print(f'Output path not provided. Defaulting to {args.output}')

"""
Data preparation
"""
out_path = args.output

print(out_path)

val_dataset = EPICtestDataset(data_root=args.EPIC_path, yaml_root=args.yaml_path, 
                            valset_yaml_root=args.valset_yaml_path, num_frames=5, repr_type=args.repr_type)
torch.autograd.set_grad_enabled(False)

val_loader = DataLoader(dataset=val_dataset, batch_size=7, shuffle=False, num_workers=4, pin_memory=True)

train_dataset = EPICDataset(data_root='./EPIC_train', yaml_root='./EPIC_train/EPIC100_state_positive_train.yaml', 
                        num_frames=config['num_frames'], repr_type=config['repr_type'])
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
model = FrameReorderNet(config).cuda().eval()
# load weights
model_weights = torch.load(config['model'])
model.load_state_dict(model_weights)

all_scores = []
all_gt = []

# Start eval
with torch.cuda.amp.autocast(enabled=not args.benchmark):
    for ti, data in tqdm(enumerate(val_loader)):  
        with torch.no_grad():
            frames = data['rgb'].cuda()
            gt_order = data['gt_order'].cuda()
            img_features = model.encode(frames) # [B, num_frames, 2048/1024]
            scores = torch.zeros(img_features.shape[0], img_features.shape[1], img_features.shape[1]).cuda() # [B, num_frames, num_frames]
            # scores[b, i,j]代表第b个batch i>j的概率
            # scores = torch.zeros((img_features.shape[0], img_features.shape[1])).cuda() # [B, num_frames],代表了每一帧的得分
            for idx1 in range(config['num_frames']):
                for idx2 in range(config['num_frames']):
                    if idx1 == idx2:
                        continue
                    else:
                        cat_feature = torch.cat([img_features[:, idx1], img_features[:, idx2]], dim = 1)
                        logits = model.classify(cat_feature) # [B,2] 
                        prob = torch.softmax(logits, dim = 1) # [B,2]
                        scores[:, idx1, idx2] = prob[:, 1] # [B]
                        # 【0，1】代表 idx1 > idx2
                        # index = torch.argmax(prob, dim = 1) # [B]
                        # scores[:,idx1] += index
            perm = get_max_permutation(scores) # B, 5
            all_scores.append(perm)
            all_gt.append(gt_order)
            
    all_scores = torch.cat(all_scores, dim = 0).cpu()
    all_gt = torch.cat(all_gt, dim = 0).cpu().numpy()
    
    print('Spearman:')
    print(np.mean([spearman_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))]))

    print('Absoulte Distance:')
    print(np.mean([absolute_distance(all_scores[i], all_gt[i]) for i in range(len(all_scores))]))

    print('Pairwise:')
    print(np.mean([pairwise_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))]))
    

print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
