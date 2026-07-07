'''
author: xin luo
create: 2026.3.4
des: loss function for model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(outputs, target, smooth=1.0):
    main_logit, aux4_logit = outputs
    main_prob = torch.sigmoid(main_logit)
    aux4_prob = torch.sigmoid(aux4_logit)
    aux4_target = F.interpolate(target, size=aux4_logit.shape[2:],  mode='area')
    main_intersection = (main_prob * target).sum(dim=(1, 2, 3))
    aux4_intersection = (aux4_prob * aux4_target).sum(dim=(1, 2, 3))
    main_union = main_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    aux4_union = aux4_prob.sum(dim=(1, 2, 3)) + aux4_target.sum(dim=(1, 2, 3))
    main_dice = (2. * main_intersection + smooth) / (main_union + smooth)
    aux4_dice = (2. * aux4_intersection + smooth) / (aux4_union + smooth)
    dice = 0.5 * main_dice + 0.5 * aux4_dice
    return 1 - dice.mean()

def deep_bce_loss(outputs, target):
    """
    outputs: (main, aux_opt, aux_nir, aux_dem), the output contains multiple head ouput. 
    target:  [B,1,H,W]    
    """
    main_logit, aux4_logit, aux3_logit, aux2_logit = outputs
    aux4_target = F.interpolate(target, size=aux4_logit.shape[2:],  mode='area')
    aux3_target = F.interpolate(target, size=aux3_logit.shape[2:],  mode='area')
    aux2_target = F.interpolate(target, size=aux2_logit.shape[2:],  mode='area')
    main_loss = bce_loss(main_logit, target)
    aux4_loss = bce_loss(aux4_logit, aux4_target)
    aux3_loss = bce_loss(aux3_logit, aux3_target)
    aux2_loss = bce_loss(aux2_logit, aux2_target)
    loss = main_loss + (aux4_loss + aux3_loss + aux2_loss)/3
    return loss

def deep_bce_dice_loss(outputs, target):
    return 0.5 * deep_bce_loss(outputs, target) + 0.5 * dice_loss(outputs, target)


def focal_loss(logit, target, alpha=0.25, gamma=2.0):
    """二分类 Focal Loss,输入 logit(未 sigmoid)"""
    prob = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pt = torch.where(target == 1, prob, 1 - prob)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = alpha_t * (1 - pt) ** gamma * bce
    return loss.mean()


def hybrid_loss(logit, target, alpha=0.25, gamma=2.0, dice_weight=0.5, smooth=1e-5):
    """Focal + Dice 混合损失,输入 logit"""
    fl = focal_loss(logit, target, alpha, gamma)
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum()
    dice = 1 - (2. * inter + smooth) / (prob.sum() + target.sum() + smooth)
    return (1 - dice_weight) * fl + dice_weight * dice