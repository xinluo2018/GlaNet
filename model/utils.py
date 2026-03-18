'''
author: xin luo
create: 2026.3.4
des: utils for model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    """Dice Loss,适合二分类和多分类分割"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + self.smooth) / \
                   (pred.sum() + target.sum() + self.smooth)
 
 
class CombinedLoss(nn.Module):
    """BCE + Dice Loss 组合（二分类常用）"""
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
 
    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight * self.bce(pred, target) +
                (1 - self.bce_weight) * self.dice(pred, target))