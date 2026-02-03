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
from utils.utils import get_lat
from utils.img2patch import crop2patch
from torchvision.transforms import v2

## create related functions
## read scene
def read_scenes(scene_paths, truth_paths, dem_paths):
  paths_zip = zip(scene_paths, truth_paths, dem_paths)
  scenes_arr = []
  truths_arr = []
  scenes_lat = []
  for scene_path, truth_path, dem_path in paths_zip:
      ## 1. read scene and truth images
      with rio.open(scene_path) as src:
        scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
        scene_lat = get_lat(src) 
      with rio.open(truth_path) as truth_src:
        truth_arr = truth_src.read(1)  # (H, W)
      ## 2. read dem
      with rio.open(dem_path) as dem_src:
        dem_arr = dem_src.read(1)  # (H, W)
      dem_arr = dem_arr[:, :, np.newaxis]  # expand to (H, W, 1)
      scene_arr = np.concatenate([scene_arr, dem_arr], axis=-1)  # (H, W, C+1)
      scenes_arr.append(scene_arr)
      truths_arr.append(truth_arr)
      scenes_lat.append(scene_lat)
  return scenes_arr, truths_arr, scenes_lat

## build custom transforms
class GaussianNoise(v2.GaussianNoise):
    def __init__(self, mean = 0.0, sigma_max=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.sigma_max = sigma_max
        self.p = p
    def transform(self, inpt, params):
        patch, ptruth = inpt[0:-1], inpt[-1:]
        if torch.rand(1) < self.p:
            self.sigma = torch.rand(1)*self.sigma_max ## update sigma        
            patch = super().transform(patch, params)
            oupt = torch.cat([patch, ptruth], dim=0)
            return oupt
        else:
            return inpt

### - Dataset definition
class ScenePathSet(torch.utils.data.Dataset):
    '''
    des: build dataset using file paths
    '''
    def __init__(self, paths_scene, paths_truth,  paths_dem=None, patch_size=(512, 512)):
        self.paths_scene = paths_scene
        self.paths_dem = paths_dem
        self.paths_truth = paths_truth
        self.patch_size = patch_size
    def __getitem__(self, idx):
        # Load scene and truth image
        scene_path = self.paths_scene[idx]
        truth_path = self.paths_truth[idx]
        ## 1. read scene and truth images
        with rio.open(scene_path) as src:
            scene_arr = src.read()  # (C, H, W)
        with rio.open(truth_path) as truth_src:
            truth_arr = truth_src.read(1)  # (H, W)
        ## 2. read dem
        if self.paths_dem is not None:
            dem_path = self.paths_dem[idx]
            with rio.open(dem_path) as dem_src:
                dem_arr = dem_src.read(1)  # (H, W)
            dem_arr = dem_arr[np.newaxis, :, :]  # expand to (1, H, W)
            scene_arr = np.concatenate([scene_arr, dem_arr], axis=0)  # (C+1, H, W)
        ## post processing

        scene_truth_arr = np.concatenate([scene_arr, truth_arr[np.newaxis, ...]], axis=0)  # (C+1, H, W)
        crop_patch = crop2patch(scene_truth_arr, channel_first=True)
        patch_truth_arr = crop_patch.toSize(size=(self.patch_size, self.patch_size))  # (c, h, w)
        patch, ptruth = patch_truth_arr[:-1, :, :], patch_truth_arr[-1:, :, :]  # (c, h, w), (1, h, w)        
        patch = torch.from_numpy(patch).float()
        ptruth = torch.from_numpy(ptruth).float()
        return patch, ptruth
    def __len__(self):
        return len(self.paths_scene)

### - Dataset definition
class SceneArraySet(torch.utils.data.Dataset):
    def __init__(self, 
                 scenes_arr, 
                 truths_arr, 
                 scenes_lat,
                 patch_size=512, 
                 patch_resize=None,
                 augment=True
                 ):
        '''
        des: build dataset using pre-loaded arrays
        args:
            scenes_arr: list of np.arrays, each array is (H, W, C) 
            truths_arr: list of np.arrays, each array is (H, W)//(C, H, W)
            scenes_lat: list of float, each is the latitude of the scene
        '''
        self.scenes_arr = scenes_arr
        self.truths_arr = truths_arr
        self.scenes_lat = scenes_lat
        self.patch_size = patch_size
        self.patch_resize = patch_resize    
        self.augment = augment    
        ## combine transforms    
        self.transforms_base = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=(patch_size, patch_size)),
        ]) 
        self.transforms_aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=180),
            GaussianNoise(mean = 0, sigma_max=0.03, p=0.3)    
        ]) 
    def __getitem__(self, idx):
        scene_arr = self.scenes_arr[idx] # (H, W, C)
        truth_arr = self.truths_arr[idx]
        scene_truth = np.concatenate([scene_arr, truth_arr[:, :, np.newaxis]], 
                                      axis=-1)
        patch_truth_ = self.transforms_base(scene_truth) 
        if self.augment:
            patch_truth_ = self.transforms_aug(patch_truth_)  ## data augmentation
        if self.patch_resize and self.patch_resize != self.patch_size:
            patch_truth_ = v2.Resize(size=(self.patch_resize, self.patch_resize))(patch_truth_)        
        patch_, ptruth_ = patch_truth_[0:-1], patch_truth_[-1:]  ## separate patch and truth        
        plat = torch.tensor(self.scenes_lat[idx]).float()
        return patch_, ptruth_, plat
    def __len__(self):
        return len(self.scenes_arr)  
    
### - Dataset definition
class PatchPathSet(torch.utils.data.Dataset):
    def __init__(self, paths_valset):
        self.paths_valset = paths_valset
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        patch_ptruth, patch_lat = torch.load(self.paths_valset[idx], weights_only=True) 
        patch_ptruth = patch_ptruth.permute(2, 0, 1)
        patch = v2.functional.to_dtype(patch_ptruth[0:-1], dtype=torch.float32) 
        ptruth = v2.functional.to_dtype(patch_ptruth[-1:], dtype=torch.float32)      
        plat = torch.tensor(patch_lat)
        return patch, ptruth, plat
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






