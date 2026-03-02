## author: xin luo, 
## created: 2021.7.8
## modify: 2021.10.13
## des: data augmentation before model training.
#       note: the 3-d np.array() is channel-first.

import torch
import random
import copy
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import v2

## build custom transforms
class GaussianNoise(v2.Transform):
    def __init__(self, mean = 0.0, sigma_max_img=0.1, sigma_max_dem=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.sigma_max_img = sigma_max_img
        self.sigma_max_dem = sigma_max_dem
        self.p = p            
    def transform(self, inpt, params):
        if torch.rand(1) < self.p:
            patch, pdem, ptruth = inpt[0:6], inpt[6:7], inpt[7:]
            self.sigma_img = torch.rand(1)*self.sigma_max_img  ## update sigma        
            self.sigma_dem = torch.rand(1)*self.sigma_max_dem  ## update sigma     
            noise_patch = torch.randn_like(patch) * self.sigma_img            
            patch = patch + noise_patch
            noise_dem = torch.randn_like(pdem) * self.sigma_dem            
            pdem = pdem + noise_dem
            inpt = torch.cat([patch, pdem, ptruth], dim=0)
        return inpt

## (delete)
# class GaussianNoise(v2.GaussianNoise):
#     def __init__(self, mean = 0.0, sigma_max=0.1, p=0.5):
#         super().__init__()
#         self.mean = mean
#         self.sigma_max = sigma_max
#         self.p = p
#     def transform(self, inpt, params):  # rewrite transform function to update sigma
#         patch, ptruth = inpt[0:-1], inpt[-1:]
#         if torch.rand(1) < self.p:
#             self.sigma = torch.rand(1)*self.sigma_max  ## update sigma        
#             patch = super().transform(patch, params)
#             oupt = torch.cat([patch, ptruth], dim=0)
#             return oupt
#         else:
#             return inpt