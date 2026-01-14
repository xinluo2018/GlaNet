import time
import torch
import random
import torch.nn as nn
from glob import glob
from notebooks import config
from utils.imgShow import imsShow
from model import unet, unet_scales 
from utils.utils import read_scenes
from utils.metrics import oa_binary, miou_binary
from utils.dataloader import SceneArraySet_scales, PatchPathSet_scales


patch_size = 256
higher_patch_size = 1024


### traset
paths_scene_tra, paths_truth_tra = config.paths_scene_tra, config.paths_truth_tra
paths_dem_tra = config.paths_dem_tra
# paths_dem_tra = config.paths_dem_adjust_tra
print(f'train scenes: {len(paths_scene_tra)}')
### valset
paths_valset = sorted(glob(f'data/dset/valset/patch_{higher_patch_size}/*'))  ## for model prediction 
# paths_patch_valset = sorted(glob(f'data/dset/valset/patch_{patch_size}_dem_adjust/*'))
print(f'vali patch: {len(paths_valset)}')


## load traset
scenes_dem_arr, truths_arr = read_scenes(paths_scene_tra, 
                                            paths_truth_tra, 
                                            paths_dem_tra) 
print('traset:', len(scenes_dem_arr))



# Create dataset instances
tra_data = SceneArraySet_scales(scenes_arr=scenes_dem_arr,    
                          truths_arr=truths_arr,   
                          patch_size=256,   
                          higher_patch_size=1024,   
                          patch_resize=True)   
val_data = PatchPathSet_scales(paths_valset=paths_valset,   
                          higher_patch_size=1024,  
                          patch_size=256,  
                          patch_resize=True)     

tra_loader = torch.utils.data.DataLoader(tra_data, 
                                         batch_size=4, 
                                         shuffle=True, 
                                         num_workers=10)
val_loader = torch.utils.data.DataLoader(val_data, 
                                         batch_size=4, 
                                         num_workers=10)

model = unet_scales(num_bands_local=7, 
                    num_bands_global=7, 
                    patch_size=patch_size,
                    higher_patch_size=higher_patch_size)

tra_loader_iter = iter(tra_loader)
val_loader_iter = iter(val_loader)
tra_one = next(tra_loader_iter)
val_one = next(val_loader_iter)
pred_local, pred_global = model(tra_one[0], tra_one[2])

### create loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                          mode='min', factor=0.6, patience=20)

loss_bce = nn.BCELoss()
def loss_rmse(y_pred, y_true):
    mse = torch.mean((y_pred - y_true)**2)
    return torch.sqrt(mse)

class Loss_scales(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_global = nn.BCELoss()
    def forward(self, x_local, y_local, x_global,  y_global):
        loss_global = self.loss_global(x_global, y_global)
        loss_local = loss_rmse(x_local, y_local)
        return loss_local + loss_global
loss_scales = Loss_scales()

'''------train step------'''
def train_step(x_patch,
               y_patch,
               x_higher_patch, 
               y_higher_patch,
               model, 
               optimizer, 
               loss_fn):
    optimizer.zero_grad()
    pred_local, pred_global = model(x_patch, x_higher_patch)
    # loss = loss_fn(pred_local, y_patch.float())
    loss = loss_fn(x_local=pred_local, 
                    y_local=y_patch.float(), 
                    x_global=pred_global, 
                    y_global=y_higher_patch.float())
    loss.backward()
    optimizer.step()    
    miou = miou_binary(pred=pred_local, truth=y_patch, device=x_patch.device)
    oa = oa_binary(pred=pred_local, truth=y_patch, device=x_patch.device)
    return loss, miou, oa
'''------validation step------'''
def val_step(x_patch,
             y_patch,
             x_higher_patch,
             y_higher_patch, 
             model,
             loss_fn):
    model.eval()
    with torch.no_grad():
        pred_local, pred_global = model(x_patch, x_higher_patch)
        # loss = loss_fn(pred_local, y_patch.float())
        loss = loss_fn(x_local=pred_local, 
                       y_local=y_patch.float(), 
                       x_global=pred_global, 
                       y_global=y_higher_patch.float())
    miou = miou_binary(pred=pred_local, truth=y_patch, device=x_patch.device)
    oa = oa_binary(pred=pred_local, truth=y_patch, device=x_patch.device)
    return loss, miou, oa

'''------train loops------'''
def train_loops(model,
                loss_fn, 
                optimizer, 
                tra_loader, 
                val_loader, 
                epoches, 
                device, 
                lr_scheduler=None):
    tra_loss_loops, tra_miou_loops, tra_oa_loops = [], [], []
    val_loss_loops, val_miou_loops, val_oa_loops = [], [], []
    model = model.to(device)
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    for epoch in range(epoches):
        start = time.time()
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0
        '''-----train the model-----'''
        for x_patch, y_patch, x_higher_patch, y_higher_patch in tra_loader:
            x_patch, y_patch, x_higher_patch, y_higher_patch = x_patch.to(device), y_patch.to(device), x_higher_patch.to(device), y_higher_patch.to(device)
            loss, miou, oa = train_step(x_patch=x_patch, 
                                        y_patch=y_patch,
                                        x_higher_patch=x_higher_patch,
                                        y_higher_patch=y_higher_patch,
                                        model=model,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn)                                        
            tra_loss += loss.item()
            tra_miou += miou.item()
            tra_oa += oa.item()
        if lr_scheduler:
            lr_scheduler.step(tra_loss)    # if using ReduceLROnPlateau
        '''----- validation the model: time consuming -----'''
        for x_patch, y_patch, x_higher_patch, y_higher_patch in val_loader:
            x_patch, y_patch, x_higher_patch, y_higher_patch = x_patch.to(device), y_patch.to(device), x_higher_patch.to(device), y_higher_patch.to(device)
            loss, miou, oa = val_step(x_patch=x_patch, 
                                    y_patch=y_patch, 
                                    x_higher_patch=x_higher_patch, 
                                    y_higher_patch=y_higher_patch, 
                                    model=model, 
                                    loss_fn=loss_fn)            
            val_loss += loss.item()
            val_miou += miou.item()
            val_oa += oa.item()
        ## Accuracy
        tra_loss = tra_loss/size_tra_loader
        val_loss = val_loss/size_val_loader
        tra_miou = tra_miou/size_tra_loader
        val_miou = val_miou/size_val_loader
        tra_oa = tra_oa/size_tra_loader
        val_oa = val_oa/size_val_loader
        tra_loss_loops.append(tra_loss); tra_miou_loops.append(tra_miou); tra_oa_loops.append(tra_oa)
        val_loss_loops.append(val_loss); val_miou_loops.append(val_miou); val_oa_loops.append(val_oa)
        print(f'Ep{epoch+1}: tra-> Loss:{tra_loss:.3f},Oa:{tra_oa:.3f},Miou:{tra_miou:.3f}, '
                f'val-> Loss:{val_loss:.3f},Oa:{val_oa:.3f}, Miou:{val_miou:.3f},time:{time.time()-start:.1f}s')
        ## show the result
        if (epoch+1)%10 == 0:
            model.eval()
            sam_index = random.randrange(len(val_data))
            patch, ptruth, higher_patch, higher_ptruth = val_data[sam_index]
            patch, ptruth = torch.unsqueeze(patch.float(), 0).to(device), ptruth.to(device)
            higher_patch, higher_ptruth = torch.unsqueeze(higher_patch.float(), 0).to(device), torch.unsqueeze(higher_ptruth.float(), 0).to(device)
            pred_local, pred_global = model(patch, higher_patch)
            ## convert to numpy and plot
            patch = patch[0].to('cpu').detach().numpy().transpose(1,2,0)
            pdem = patch[:,:, -1]
            pred_patch = pred_local[0].to('cpu').detach().numpy()
            ptruth = ptruth.to('cpu').detach().numpy()
            pred_higher_patch = pred_global[0].to('cpu').detach().numpy()
            higher_patch = higher_patch[0].to('cpu').detach().numpy().transpose(1,2,0)
            higher_ptruth = higher_ptruth.to('cpu').detach().numpy()
            imsShow([higher_patch, pred_higher_patch, patch, pdem, pred_patch, ptruth], 
                    clip_list = (2,0,2,2,0,0),
                    img_name_list=['input_higher_patch', 'pred_higher_patch', 'input_patch', 
                                   'pdem', 'prediction', 'truth'],                     
                    figsize=(15,3))
    metrics = {'tra_loss':tra_loss_loops, 'tra_miou':tra_miou_loops, 'tra_oa': tra_oa_loops, 
                    'val_loss': val_loss_loops, 'val_miou': val_miou_loops, 'val_oa': val_oa_loops}
    return metrics 


if __name__ == '__main__':
  device = torch.device('cuda:0')  
  metrics = train_loops(model=model,  
                        epoches=200,  
                        loss_fn=loss_scales,  
                        optimizer=optimizer,  
                        lr_scheduler=lr_scheduler,   
                        tra_loader=tra_loader,   
                        val_loader=val_loader,  
                        device=device)  


