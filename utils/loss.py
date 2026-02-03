'''
author: xin luo
create: 2025-12-29
des: loss functions for segmentation tasks
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=3.0, reduction='mean', eps=1e-8):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_prob, target_prob):
        pred_prob = torch.clamp(pred_prob, self.eps, 1 - self.eps)        
        bce = - (target_prob * torch.log(pred_prob) + 
                  (1 - target_prob) * torch.log(1 - pred_prob))
        
        pt = torch.where(target_prob == 1, pred_prob, 1 - pred_prob)
        modulating_factor = (1 - pt) ** self.gamma
        alpha_factor = torch.where(target_prob == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * modulating_factor * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class HybridLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = BinaryFocalLoss(alpha, gamma, reduction='none')
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        
        # Dice系数计算
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = 1 - (2. * intersection + 1e-5) / (union + 1e-5)
        
        # 混合损失
        loss = (1 - self.dice_weight) * focal_loss.mean() + self.dice_weight * dice
        return loss




