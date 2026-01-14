''' 
author: xin luo
create: 2025.12.20
des: dataset and dataloader for deep learning tasks in remote sensing
'''

import cv2 
import random
import torch
import rasterio as rio
import numpy as np
from utils.img2patch import crop2patch

## create related functions
## - crop scene to patches
class RandomCrop:
    '''
    des: randomly crop corresponding to specific patch size
    '''
    def __init__(self, size=(256, 256)):
        self.size = size
        self.loc_start = []
    def __call__(self, image, truth):
        '''
        image: np.array(), channel_first (C, H, W)
        truth: np.array(), (H, W)/(C, H, W)
        '''
        if len(truth.shape)==2:
            truth = truth[np.newaxis, ...]
        start_row = random.randint(0, truth.shape[1]-self.size[0])
        start_col = random.randint(0, truth.shape[2]-self.size[1])
        patch = image[:, start_row:start_row+self.size[0],start_col:start_col+self.size[1]]
        ptruth = truth[:, start_row:start_row+self.size[0], start_col:start_col+self.size[1]]
        self.loc_start = [start_row, start_col]
        return patch, ptruth
    
### - Dataset definition
class ScenePathSet(torch.utils.data.Dataset):
    '''
    des: build dataset using file paths
    '''
    def __init__(self, paths_scene, paths_truth,  paths_dem=None, path_size=(512, 512)):
        self.paths_scene = paths_scene
        self.paths_dem = paths_dem
        self.paths_truth = paths_truth
        self.path_size = path_size
    def __getitem__(self, idx):
        # Load scene and truth image
        scene_path = self.paths_scene[idx]
        truth_path = self.paths_truth[idx]
        ## 1. read scene and truth images
        with rio.open(scene_path) as src:
            scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
        with rio.open(truth_path) as truth_src:
            truth_arr = truth_src.read(1)  # (H, W)
        ## 2. read dem
        if self.paths_dem is not None:
            dem_path = self.paths_dem[idx]
            with rio.open(dem_path) as dem_src:
                dem_arr = dem_src.read(1)  # (H, W)
            dem_arr = dem_arr[:, :, np.newaxis]  # expand to (H, W, 1)
            scene_arr = np.concatenate([scene_arr, dem_arr], axis=-1)  # (H, W, C+1)
        ## post processing
        scene_arr = scene_arr.astype(np.float32).transpose((2, 0, 1)) # (C, H, W)
        patch, truth = RandomCrop(size=(self.path_size, self.path_size))(scene_arr, truth_arr)  # crop
        truth = truth[np.newaxis, :].astype(np.int8)  # (C, H, W)
        patch = torch.from_numpy(patch).float()
        truth = torch.from_numpy(truth).float()
        return patch, truth
    def __len__(self):
        return len(self.paths_scene)

### - Dataset definition
class SceneArraySet(torch.utils.data.Dataset):
    def __init__(self, scenes_arr, truths_arr, patch_size=512, patch_resize=None):
        '''
        des: build dataset using pre-loaded arrays
        args:
            scenes_arr: list of np.arrays, each array is (H, W, C) 
            truths_arr: list of np.arrays, each array is (H, W)//(C, H, W)
            path_size: tuple, (height, width) of cropped patch
        '''
        self.scenes_arr = scenes_arr
        self.truths_arr = truths_arr
        self.patch_size = patch_size
        self.patch_resize = patch_resize
    def __getitem__(self, idx):
        scene_arr = self.scenes_arr[idx].astype(np.float32).transpose((2, 0, 1)) # (C, H, W)
        truth_arr = self.truths_arr[idx]
        patch, ptruth = RandomCrop(size=(self.patch_size, self.patch_size))(scene_arr, truth_arr)  # crop
        patch, ptruth = patch.transpose((1,2,0)), ptruth.transpose((1,2,0))  # to (H, W, C)
        if self.patch_resize is not None:
            patch = cv2.resize(patch, dsize=(self.patch_resize, self.patch_resize), interpolation=cv2.INTER_AREA)
            ptruth = cv2.resize(ptruth, dsize=(self.patch_resize, self.patch_resize), interpolation=cv2.INTER_NEAREST)
            ptruth = ptruth[..., np.newaxis]        
        patch, ptruth = patch.transpose((2,0,1)), ptruth.transpose((2,0,1))  # to (C, H, W)
        patch = torch.from_numpy(patch).float()
        ptruth = torch.from_numpy(ptruth).float()
        return patch, ptruth
    def __len__(self):
        return len(self.scenes_arr) 

### - Dataset definition
class PatchPathSet(torch.utils.data.Dataset):
    def __init__(self, paths_valset):
        self.paths_valset = paths_valset
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        patch_ptruth = torch.load(self.paths_valset[idx], weights_only=True) 
        patch_ptruth = patch_ptruth.permute(2,0,1)  # (C, H, W)
        return patch_ptruth[0:-1], patch_ptruth[-1:]
    def __len__(self):
        return len(self.paths_valset)

### - Dataset definition
class PatchPathSet_2(torch.utils.data.Dataset):
    def __init__(self, paths_valset, patch_resize=None):
        self.paths_valset = paths_valset
        self.patch_resize = patch_resize
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        patch_ptruth = torch.load(self.paths_valset[idx], weights_only=True) 
        patch, ptruth = patch_ptruth[...,0:-1], patch_ptruth[...,-1:]        
        ## crop inner 256x256 for validation
        if ptruth.shape[0]>256:
            crop_start = (patch.shape[0]-256)//2
            ptruth = ptruth[crop_start:crop_start+256, 
                            crop_start:crop_start+256, :]
        if self.patch_resize is not None:
            patch = cv2.resize(patch.numpy().astype(np.float32), 
                               dsize=(self.patch_resize, self.patch_resize), 
                               interpolation=cv2.INTER_AREA)
            patch = torch.from_numpy(patch).float()
        patch, ptruth = patch.permute((2,0,1)), ptruth.permute((2,0,1))  ## to (C, H, W)
        return patch, ptruth
    def __len__(self):
        return len(self.paths_valset)

### - Dataset definition

class SceneArraySet_scales(torch.utils.data.Dataset):
    def __init__(self, scenes_arr, 
                 truths_arr, 
                 patch_size=256,
                 higher_patch_size=1024,
                 patch_resize=True):
        '''
        des: build dataset using pre-loaded arrays
        args:
            scenes_arr: list of np.arrays, each array is (H, W, C) 
            truths_arr: list of np.arrays, each array is (H, W)
            patch_size: int, size of cropped patch
            higher_patch_size: int, size of higher scale cropped patch
        '''
        self.scenes_arr = scenes_arr
        self.truths_arr = truths_arr
        self.patch_size = patch_size
        self.higher_patch_size = higher_patch_size
        self.patch_resize = patch_resize
    def __getitem__(self, idx):
        scene_arr = self.scenes_arr[idx].astype(np.float32)  # (H, W, C)
        struth_arr = self.truths_arr[idx][...,np.newaxis]  # (H, W, 1)
        scene_struth_arr = np.concatenate([scene_arr, struth_arr], axis=-1)  # (H, W, C+1)
        scene_truth_crop = crop2patch(scene_struth_arr, channel_first=False)
        [higher_patch_ptruth, patch_ptruth] = scene_truth_crop.toScales(scales=(self.higher_patch_size, self.patch_size),
                                                                        resize=self.patch_resize)
        ## higher patch
        higher_patch, higher_ptruth = higher_patch_ptruth[:,:,0:-1], higher_patch_ptruth[:,:,-1:]
        higher_patch = torch.from_numpy(higher_patch).float().permute(2,0,1) # to (C, H, W)
        higher_ptruth = torch.from_numpy(higher_ptruth).float().permute(2,0,1)  # to (C, H, W)
        ## patch
        patch_ptruth = torch.from_numpy(patch_ptruth).float().permute(2,0,1)  # (C+1, H, W)        
        patch, ptruth = patch_ptruth[0:-1,:,:], patch_ptruth[-1:,:,:]
        return patch, ptruth, higher_patch, higher_ptruth
    def __len__(self):
        return len(self.scenes_arr)

class PatchPathSet_scales(torch.utils.data.Dataset):
    def __init__(self, paths_valset, higher_patch_size, 
                                patch_size=256, patch_resize=True):
        self.paths_valset = paths_valset
        self.patch_size = patch_size
        self.higher_patch_size = higher_patch_size
        self.patch_resize = patch_resize
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        higher_patch_ptruth = torch.load(self.paths_valset[idx], weights_only=True) 
        higher_patch, higher_ptruth = higher_patch_ptruth[...,0:-1], higher_patch_ptruth[...,-1:]        
        ## get local 256 patch (for validation)'
        crop_start = (higher_patch.shape[0]-256)//2
        patch = higher_patch[crop_start:crop_start+256, 
                                        crop_start:crop_start+256, :]   
        ptruth = higher_ptruth[crop_start:crop_start+256, 
                                        crop_start:crop_start+256, :]      
        patch, ptruth = patch.permute((2,0,1)), ptruth.permute((2,0,1))  # to (C, H, W)  
        if self.patch_resize:
            higher_patch = cv2.resize(higher_patch.numpy().astype(np.float32), 
                               dsize=(256, 256), 
                               interpolation=cv2.INTER_AREA)
            higher_patch = torch.from_numpy(higher_patch).float()
            higher_ptruth = cv2.resize(higher_ptruth.numpy().astype(np.float32), 
                               dsize=(256, 256), 
                               interpolation=cv2.INTER_NEAREST)
            higher_ptruth = torch.from_numpy(higher_ptruth[...,np.newaxis]).float()    
        higher_patch = higher_patch.permute((2,0,1))   # to (C, H, W)
        higher_ptruth = higher_ptruth.permute((2,0,1))  # to (C, H, W)
        return patch, ptruth, higher_patch, higher_ptruth
    def __len__(self):
        return len(self.paths_valset)

class PatchArraySet_scales(torch.utils.data.Dataset):
    def __init__(self, patch_arr_list, higher_patch_arr_list):
        self.patch_arr_list = patch_arr_list
        self.higher_patch_arr_list = higher_patch_arr_list
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        patch_ptruth = self.patch_arr_list[idx]
        higher_patch_ptruth = self.higher_patch_arr_list[idx]
        patch_ptruth = patch_ptruth.permute(2,0,1)  # (C, H, W)
        higher_patch_ptruth = higher_patch_ptruth.permute(2,0,1)  # (C, H, W)
        patch, ptruth = patch_ptruth[0:-1,:,:], patch_ptruth[-1:,:,:]
        higher_patch, higher_ptruth = higher_patch_ptruth[0:-1,:,:], higher_patch_ptruth[-1:,:,:]
        return patch, ptruth, higher_patch, higher_ptruth
    def __len__(self):
        return len(self.patch_arr_list)






