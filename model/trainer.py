import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import git
import datetime
# TODO change to relative path
sys.path.append('/home/venom/projects/XMem/')
# from model.losses import LossComputer
# from network import XMem
from copy import deepcopy

import model.resnet as resnet
from model.network import FrameReorderNet
from model.losses import LossComputer
import matplotlib.pyplot as plt
from util.log_integrator import Integrator
from tqdm import tqdm
import scipy
from scipy import stats

def spearman_acc(story, gt_order):
    return scipy.stats.spearmanr(story, gt_order)[0]

def absolute_distance(story, gt_order):
    return np.mean(np.abs(np.array(story) - np.array(gt_order)))

def pairwise_acc(story, gt_order):
    correct = 0
    index = np.argsort(gt_order)
    story = list(np.array(story)[index]) # 这样gtorder就变成01234了，相当于交换了一下
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            if story[idx1] < story[idx2]:
                correct += 1
    return correct/total

def validate(model, val_loader):
    all_scores = []
    all_gt = []
    # Start eval
    for ti, data in tqdm(enumerate(val_loader)):  
        with torch.no_grad():
            frames = data['rgb'].cuda()
            gt_order = data['gt_order'] # B,5
            img_features = model.encode(frames) # [B, num_frames, 2048/1024]
            scores = torch.zeros((img_features.shape[0], img_features.shape[1])).cuda() # [B, num_frames],代表了每一帧的得分
            for idx1 in range(5):
                for idx2 in range(5):
                    if idx1 == idx2:
                        continue
                    else:
                        cat_feature = torch.cat([img_features[:, idx1], img_features[:, idx2]], dim = 1)
                        logits = model.classify(cat_feature) # [B,2] 
                        # 【0，1】代表 idx1 > idx2
                        index = torch.argmax(logits, dim = 1) # [B]
                        scores[:,idx1] += index
            all_scores.append(scores)
            all_gt.append(gt_order)
    
    all_scores = torch.cat(all_scores, dim = 0).cpu()
    all_gt = torch.cat(all_gt, dim = 0).numpy()
    # print(f"gt_shape:{all_gt}")
    # print(f"all_scores: {all_scores[:10]}")
    all_scores = torch.argsort(all_scores, dim = 1).numpy()
    # print(f"all_scores: {all_scores[:10]}")

    Spearman = np.mean([spearman_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))])

    Absoulte_Distance = np.mean([absolute_distance(all_scores[i], all_gt[i]) for i in range(len(all_scores))])

    Pairwise = np.mean([pairwise_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))])
    
    return {'Spearman':Spearman, 
            'Absoulte_Distance':Absoulte_Distance, 
            'Pairwise':Pairwise}

class Trainer:
    def __init__(self, config, logger, local_rank, world_size):
        self.config = config
        self.logger = logger
        
        network = FrameReorderNet(config=config)
        self.model = nn.parallel.DistributedDataParallel(
                    network.cuda(), 
                    device_ids=[local_rank], output_device=local_rank, 
                    broadcast_buffers=False, find_unused_parameters=True)
        # 升级版的average_meter 
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)
        self.train()
        
        if logger is not None:
            self.last_time = time.time()
            self.logger.log(f'model_size:{str(sum([param.nelement() for param in self.model.parameters()]))}')
            self.save_path = logger._save_dir
        else:
            self.save_path = None
        
        if self.config['freeze']:
            print('Freezing the encoder in Trainer')
            for param in self.model.module.Encoder.parameters():
                param.requires_grad = False
        
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        
        print('parameter not requires grad:')
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
        print('------------------')

        
        if config['cos_lr']:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config['iterations'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
            
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.save_network_interval = config['save_network_interval']
        
    def do_pass(self, data, it, val_loader):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        frames = data['rgb'].cuda() # [B, num_frames, 3, H, W]
        gt_order = data['gt_order'].cuda() # [B, num_frames]
        img_features = self.model.module.encode(frames) # [B, num_frames, 2048/1024/768]
        # print(f"img_features:{img_features.shape}")
        all_logits = []
        all_target = []
        for idx1 in range(self.config['num_frames']):
            for idx2 in range(self.config['num_frames']):
                if idx1 == idx2:
                    continue
                else:
                    cat_feature = torch.cat([img_features[:, idx1], img_features[:, idx2]], dim = 1)
                    logits = self.model.module.classify(cat_feature) # [B,2]
                    target = (gt_order[:, idx1] > gt_order[:, idx2]).to(torch.int64) # [B]
                    
                    all_logits.append(logits)
                    all_target.append(target)
        print(f"info:{data['info']}")
        print(f"all_target:{torch.stack(all_target, 0)}")
        print(f"all_logits:{torch.stack(all_logits, 0)}")
        # dd
        all_logits = torch.stack(all_logits, 0).flatten(start_dim=0, end_dim=1)
        all_target = torch.stack(all_target, 0).flatten(start_dim=0, end_dim=1)
        print(f"all_target:{all_target}")
        print(f"all_logits:{all_logits}")
        losses = self.loss_computer.compute(all_logits, all_target)
        
        # recording
        if self.logger is not None:
            self.integrator.add_dict(losses)
        
        if self._is_train:
            if (it) % self.log_text_interval == 0 and it != 0:
                train_metrics = self.train_integrator.finalize()
                
                if self.logger is not None:
                    eval_metrics = validate(self.model.module, val_loader)
                    self.logger.write(prefix='reorder', train_metrics=train_metrics, eval_metrics=eval_metrics,**{'lr':self.scheduler.get_last_lr()[0],
                                    'time':(time.time()-self.last_time)/self.log_text_interval})
                    all_dicts = {**train_metrics, **{'lr':self.scheduler.get_last_lr()[0],
                                        'time':(time.time()-self.last_time)/self.log_text_interval}}
                    self.last_time = time.time()
                    for k, v in all_dicts.items():
                        msg = 'It {:6d} [{:5s}] [{:13}]: {:s}'.format(it, 'TRAIN', k, '{:.9s}'.format('{:0.9f}'.format(v)))
                        if self.logger is not None:
                            self.logger.log(msg)
                    for k, v in eval_metrics.items():
                        msg = 'It {:6d} [{:5s}] [{:13}]: {:s}'.format(it, 'EVAL', k, '{:.9s}'.format('{:0.9f}'.format(v)))
                        if self.logger is not None:
                            self.logger.log(msg)
                    print('-------------------')
                self.train_integrator.reset_except_hooks()

            if it % self.save_network_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save_network(it)
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()
        self.scheduler.step()
        
        # print(f"all_logits:{all_logits.shape}")
        # print(f"all_target:{all_target.shape}")
        # print(f"all_targets:{all_target}")

    
    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # 不使用BN和dropout
        self.model.train()
        return self

    def eval(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self
    
    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}/network_{it}.pth'
        torch.save(self.model.module.state_dict(), model_path)
        torch.save(self.model.module.state_dict(), f'{self.save_path}/latest_network.pth')
        print(f'Network saved to {model_path}.')