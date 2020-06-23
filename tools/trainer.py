# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import FullSampler

class Trainer (nn.Module):
    """ Helper class to train a deep network.
        Overload this class `forward_backward` for your actual needs.
    
    Usage: 
        train = Trainer(net, loader, loss, optimizer)
        for epoch in range(n_epochs):
            train()
    """
    def __init__(self, net, train_loader, val_loader, loss, optimizer, scheduler=None):
        nn.Module.__init__(self)
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def iscuda(self):
        return next(self.net.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k:self.todevice(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)):
            return [self.todevice(v)  for v in x]
        
        if self.iscuda(): 
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def _forward_backward(self, inputs):
        output = self.net(imgs=[inputs.pop('img1'), inputs.pop('img2')])
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details

    def train(self):
        self.net.train()
        
        stats = defaultdict(list)
        
        for iter, inputs in enumerate(tqdm(self.train_loader)):
            inputs = self.todevice(inputs)
            '''
            # visualize images
            plt.figure(figsize=(12, 8))
            for img1, img2, aflow in zip(inputs["img1"], inputs["img2"], inputs["aflow"]):
                grid = FullSampler._aflow_to_grid(aflow.unsqueeze(0))
                img2_w = F.grid_sample(img2.unsqueeze(0).permute(0, 3, 1, 2).float(), grid, align_corners=False)
                print(grid.shape, img1.shape, img2.shape, img2_w.shape)

                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.imshow(img1.cpu().numpy())
                plt.tight_layout()

                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.imshow(img2.cpu().numpy())
                plt.tight_layout()

                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.imshow(img2_w.squeeze().int().permute(1, 2, 0).cpu().numpy())
                plt.tight_layout()

                plt.show()
            '''
            
            # compute gradient and do model update
            self.optimizer.zero_grad()
            
            loss, details = self._forward_backward(inputs)
            if torch.isnan(loss):
                raise RuntimeError('Loss is NaN')
            
            self.optimizer.step()
            
            for key, val in details.items():
                stats[key].append( val )

        if self.scheduler is not None:
            self.scheduler.step()

        print(" Summary of losses during this epoch:")
        mean = lambda lis: sum(lis) / len(lis)
        for loss_name, vals in stats.items():
            N = 1 + len(vals)//10
            print(f"  - {loss_name:20}:", end='')
            print(f" {mean(vals[:N]):.3f} --> {mean(vals[-N:]):.3f} (avg: {mean(vals):.3f})")
        return mean(stats['loss']) # return average loss

    def validate(self):
        self.net.eval()
        stats = defaultdict(list)

        with torch.no_grad():
            for _, inputs in enumerate(tqdm(self.val_loader)):
                inputs = self.todevice(inputs)
                output = self.net(imgs=[inputs.pop('img1'), inputs.pop('img2')])
                allvars = dict(inputs, **output)
                loss, details = self.loss_func(**allvars)
                if torch.isnan(loss):
                    raise RuntimeError('Loss is NaN')
                for key, val in details.items():
                    stats[key].append(val)

            print(" Summary of losses on validation data:")
            mean = lambda lis: sum(lis) / len(lis)
            for loss_name, vals in stats.items():
                N = 1 + len(vals) // 10
                print(f"  - {loss_name:20}:", end='')
                print(f" {mean(vals[:N]):.3f} --> {mean(vals[-N:]):.3f} (avg: {mean(vals):.3f})")
            return mean(stats['loss'])  # return average loss
