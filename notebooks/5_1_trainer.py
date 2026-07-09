'''
author: xin luo
create: 2026.5.17
des: training script for swin-based model
'''

import sys 
sys.path.append('/home/ps/Develop/dev-luo/GlaNet')  ## add the current working directory to sys.path for module import
import time
import torch
import pandas as pd
import torch.nn as nn
from glob import glob
import torch.nn.functional as F
try: from notebooks import config
except ModuleNotFoundError: import config
from torchvision.transforms import v2
from utils.data_aug import GaussianNoise
from utils.dataloader import read_scenes 
from utils.dataloader import SceneArraySet, PatchPathSet
from model import u3net_cross_fusion, u2net_timm, u3net_timm_, u3net_timm, unet_timm
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy

## 1. params 
patch_size = 512  ## patch size setting
patch_resize = None  ## patch resize setting
learning_rate = 1e-4
batch_size_tra = 8
batch_size_val = 16 
device = torch.device('cuda:1')   
# path_pretrained = 'model/trained/u3net_cross_fusion_.pth'   ## pretrained model path, if None, train from scratch
# path_pretrained = 'model/trained/ablation_module/u3net_timm_09537.pth'             ## pretrained model path, if None, train from scratch
model_name = 'u3net_timm_rgb_remove'  ## model name for saving

### traset
paths_scene_tra, paths_truth_tra = config.paths_scene_tra, config.paths_truth_tra
paths_dem_tra = config.paths_dem_tra
print(f'train scenes: {len(paths_scene_tra)}')

## valset
paths_valset = sorted(glob(f'data/dset/valset/patch_{patch_size}/*.pt'))  ## for model prediction 
print(f'vali patch {patch_size}: {len(paths_valset)}')
path_valset_lat = f'data/dset/valset/patch_{patch_size}/patches_lat.json'  ## for model prediction

## 2. Read data
scenes_arr, truths_arr, scenes_lat = read_scenes(paths_scene_tra, paths_truth_tra, paths_dem_tra, lat=True)    # type: ignore

## 3. dataloader
transforms_tra = v2.Compose([   
            v2.ToImage(),   
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomVerticalFlip(p=0.5),
            v2.RandomCrop(size=(patch_size, patch_size)),   
            v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),   # type:ignore
            GaussianNoise(mean = 0.0, sigma_max_img=0.1, sigma_max_dem=0, p=0.3)
            ]) 
transforms_val = v2.Compose([v2.ToDtype(torch.float32)])   

# Create dataset instances
tra_data = SceneArraySet(scenes_arr=scenes_arr, truths_arr=truths_arr, 
                              patch_size=patch_size, transforms=transforms_tra, scenes_lat=scenes_lat)
val_data = PatchPathSet(paths_valset=paths_valset, transforms=transforms_val, path_valset_lat=path_valset_lat)

## Create data loaders
tra_loader = torch.utils.data.DataLoader(tra_data, batch_size=batch_size_tra, 
                                         shuffle=True, num_workers=5)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size_val, num_workers=5)

## 4. model, loss and optimizer
# model = unet_timm(# backbone_name='resnet34', 
#                     backbone_name='efficientnet_b0',
#                     pretrained=True)
# model = u2net_timm(# backbone_name='resnet34', 
#                     backbone_name='efficientnet_b0',
#                     pretrained=True)
# model = u3net_timm(# backbone_name='resnet34', 
#                     backbone_name='efficientnet_b0',
#                     pretrained=True)
model = u3net_timm_(
                    # backbone_name='resnet34', 
                    backbone_name='efficientnet_b0',
                    pretrained=True)
# model = u3net_cross_fusion(
#                     # backbone_name='resnet50', 
#                     backbone_name='efficientnet_b0',
#                     pretrained=False)  

# # 4.1 load pretrained weights
# model_dict = model.state_dict() 
# model_checkpoint = torch.load(path_pretrained, map_location='cpu') 
# pretrained_dict = {
#     k: v for k, v in model_checkpoint.items() 
#     if k in model_dict and v.shape == model_dict[k].shape}
# discarded_keys = set(model_checkpoint.keys()) - set(pretrained_dict.keys())
# if discarded_keys:
#     print(f"unmatched parameters and are discarded:")
#     print('='*50)
#     for k in sorted(discarded_keys):
#         print(f"  - {k}")
#     print('='*50)
# else:
#     print('no discarded parameters')
# newly_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
# if newly_keys:
#     print(f"newly added parameters and are randomly initialized:")
#     print('='*50)
#     for k in sorted(newly_keys):
#         print(f"  - {k}")
#     print('='*50)
# else:
#     print("no newly added parameters")
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

### create loss and optimizer  
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  ## all params
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)  ## only trainable (non-frozen) params

## 5. train and val loops
'''------train loops------'''
def train_loops(model, loss_fn, 
                    optimizer, 
                    tra_loader, 
                    val_loader,  
                    epoches, 
                    device, 
                    lr_scheduler=None):
    loss_tra_loops, miou_tra_loops, oa_tra_loops = [], [], []
    loss_val_loops, miou_val_loops, oa_val_loops = [], [], []
    model = model.to(device)
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    best_miou = 0.80
    epoches_i = []
    for epoch in range(epoches):
        start = time.time()
        loss_tra, loss_val = 0, 0
        '''-----train the model-----'''
        miou_tra = BinaryJaccardIndex().to(device)
        oa_tra = BinaryAccuracy().to(device)
        model.train()   # training mode for dropout and batchnorm
        # freeze_encoders()   # re-force frozen encoders back to eval (model.train() reactivates their BN)
        for x_batch, y_batch, lat_batch in tra_loader:
            x_batch, y_batch, lat_batch = x_batch.to(device), y_batch.to(device), lat_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            pred = (F.sigmoid(preds[0]) > 0.5).float()
            miou_tra.update(pred, y_batch.long())
            oa_tra.update(pred, y_batch.long())
            loss_tra += loss.item()
        miou_tra_global = miou_tra.compute()
        oa_tra_global = oa_tra.compute()
        loss_tra_global = loss_tra/size_tra_loader
        miou_tra.reset(); oa_tra.reset()

        '''----- validation the model: time consuming -----'''
        oa_val = BinaryAccuracy().to(device)
        miou_val = BinaryJaccardIndex().to(device)
        model.eval()
        if epoch > 1000 and (epoch+1) % 2 == 0: 
            for x_batch, y_batch, lat_batch in val_loader:
                x_batch, y_batch, lat_batch = x_batch.to(device), y_batch.to(device), lat_batch.to(device)
                with torch.no_grad():
                    preds = model(x_batch)
                    loss = loss_fn(preds, y_batch)
                pred = (F.sigmoid(preds[0]) > 0.5).float()
                miou_val.update(pred, y_batch.long())
                oa_val.update(pred, y_batch.long())
                loss_val += loss.item()
            miou_val_global = miou_val.compute()
            oa_val_global = oa_val.compute()
            loss_val_global = loss_val/size_val_loader
            miou_val.reset(); oa_val.reset()
            loss_tra_loops.append(loss_tra_global); miou_tra_loops.append(miou_tra_global.item()); oa_tra_loops.append(oa_tra_global.item())
            loss_val_loops.append(loss_val_global); miou_val_loops.append(miou_val_global.item()); oa_val_loops.append(oa_val_global.item())
            epoches_i.append(epoch)
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, '
                    f'val-> Loss:{loss_val_global:.4f},Oa:{oa_val_global:.4f}, Miou:{miou_val_global:.4f},time:{time.time()-start:.1f}s')
            ## save the best model
            if miou_val_global.item() > best_miou:
                best_miou = miou_val_global.item()       ## update best miou
                torch.save(model.state_dict(), f'model/trained/ablation_data/{model_name}_0{str(round(best_miou*10000))}.pth')
        else: 
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, \
                                time:{time.time()-start:.1f}s')
        if lr_scheduler:
          lr_scheduler.step(miou_tra_global)    ## if use lr_scheduler like ReduceLROnPlateau

    metrics = {'epoch':epoches_i, 'tra_loss':loss_tra_loops, 'tra_oa': oa_tra_loops, 'tra_miou': miou_tra_loops,
                'val_loss': loss_val_loops, 'val_oa': oa_val_loops, 'val_miou': miou_val_loops}
    return metrics 

if __name__ == '__main__':
    metrics = train_loops(model=model,  
                    epoches=3000,   
                    # loss_fn=bce_loss,   
                    loss_fn = deep_bce_loss,
                    optimizer=optimizer,  
                    tra_loader=tra_loader,   
                    val_loader=val_loader,   
                    # lr_scheduler=lr_scheduler,   
                    device=device)

    torch.save(model.state_dict(), f'model/trained/ablation_data/{model_name}_.pth')
    ## metrics saving
    path_metrics = f'training_metrics.csv'    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(path_metrics, index=False, sep=',')

