'''
author: xin luo
create: 2025.12.21
des: U-Net model for multi-scale learning
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import unet




def batch_global2local(batch_data,
                       patch_size=512,
                       higher_patch_size=1024):
    crop_start = (higher_patch_size - patch_size) // 2
    batch_upsam = F.interpolate(batch_data, 
                                size=(higher_patch_size, higher_patch_size), 
                                mode='bilinear',
                                align_corners=True)
    batch_upsam_crop = batch_upsam[:,:, 
                                    crop_start:crop_start+patch_size, 
                                    crop_start:crop_start+patch_size]
    return batch_upsam_crop

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class unet_local(nn.Module):
    def __init__(self, num_bands):
        super(unet_local, self).__init__()
        self.num_bands = num_bands
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_conv1 = conv3x3_bn_relu(self.num_bands, 16)
        self.down_conv2 = conv3x3_bn_relu(16, 32)
        self.down_conv3 = conv3x3_bn_relu(32, 64)
        self.down_conv4 = conv3x3_bn_relu(64, 128)
        self.up_conv1 = conv3x3_bn_relu(192, 64)
        self.up_conv2 = conv3x3_bn_relu(96, 48)
        self.up_conv3 = conv3x3_bn_relu(64, 32)
        self.outlayer_global = nn.Sequential(
                                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                nn.Sigmoid()
                                ) 

    def forward(self, x):   ## input size: 6x256x256
        ## encoder part
        x1 = self.down_conv1(x)              
        x1 = F.avg_pool2d(input=x1, kernel_size=2)  # 16x128x128
        x2 = self.down_conv2(x1)              
        x2 = F.avg_pool2d(input=x2, kernel_size=2) # 32x64x64
        x3 = self.down_conv3(x2)              
        x3 = F.avg_pool2d(input=x3, kernel_size=2) # 64x32x32
        x4 = self.down_conv4(x3)              
        x4 = F.avg_pool2d(input=x4, kernel_size=2) # 128x16x16
        ## decoder part
        x4_up = torch.cat([self.up(x4), x3], dim=1)  # (128+64)x32x32
        x3_up = self.up_conv1(x4_up)  # 64x32x32
        x3_up = torch.cat([self.up(x3_up), x2], dim=1)  # (64+32)x64x64
        x2_up = self.up_conv2(x3_up)  # 48x64x64
        x2_up = torch.cat([self.up(x2_up), x1], dim=1)  # (48+16)x128x128
        x1_up = self.up_conv3(x2_up)    # 32x128x128
        x1_up = self.up(x1_up)        # 32x256x256 
        pro_pred = self.outlayer_global(x1_up)
        return pro_pred          

class unet_global(nn.Module):
    def __init__(self, num_bands):
        super(unet_global, self).__init__()
        self.num_bands = num_bands
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_conv1 = conv3x3_bn_relu(self.num_bands, 16)
        self.down_conv2 = conv3x3_bn_relu(16, 32)
        self.down_conv3 = conv3x3_bn_relu(32, 64)
        self.down_conv4 = conv3x3_bn_relu(64, 128)
        self.up_conv1 = conv3x3_bn_relu(192, 64)
        self.up_conv2 = conv3x3_bn_relu(96, 48)
        self.up_conv3 = conv3x3_bn_relu(64, 32)
        self.outlayer_global = nn.Sequential(
                                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                nn.Sigmoid()
                                ) 
    def forward(self, x):   ## input size: 6x256x256 (take for example)
        ## encoder part
        x1 = self.down_conv1(x)              
        x1 = F.avg_pool2d(input=x1, kernel_size=2)  # 16x128x128
        x2 = self.down_conv2(x1)              
        x2 = F.avg_pool2d(input=x2, kernel_size=2) # 32x64x64
        x3 = self.down_conv3(x2)              
        x3 = F.avg_pool2d(input=x3, kernel_size=2) # 64x32x32
        x4 = self.down_conv4(x3)              
        x4 = F.avg_pool2d(input=x4, kernel_size=2) # 128x16x16
        ## decoder part
        x4_up = torch.cat([self.up(x4), x3], dim=1)  # (128+64)x32x32
        x3_up = self.up_conv1(x4_up)  # 64x32x32
        x3_up = torch.cat([self.up(x3_up), x2], dim=1)  # (64+32)x64x64
        x2_up = self.up_conv2(x3_up)  # 48x64x64
        x2_up = torch.cat([self.up(x2_up), x1], dim=1)  # (48+16)x128x128
        x1_up = self.up_conv3(x2_up)    # 32x128x128
        x1_up = self.up(x1_up)        # 32x256x256
        pro_pred = self.outlayer_global(x1_up)
        return pro_pred          

class unet_scales(nn.Module):
    def __init__(self, num_bands_local, 
                    num_bands_global, 
                    patch_size=512,
                    higher_patch_size=1024):
        super(unet_scales, self).__init__()
        self.patch_size = patch_size        
        self.higher_patch_size = higher_patch_size
        self.model_local = unet_local(num_bands_local+1)
        self.model_global = unet_global(num_bands=num_bands_global)
        self.model_global = unet(num_bands=7)
        path_model_global = 'model/trained/patch_1024/unet_resize256_weights.pth'
        self.model_global.load_state_dict(torch.load(path_model_global, weights_only=True))  # load the weights of the trained model

    def forward(self, x_local, x_global):    
        prob_global = self.model_global(x_global)  
        prob_global2local = batch_global2local(batch_data=prob_global, 
                                                patch_size=self.patch_size,
                                                higher_patch_size=self.higher_patch_size)
        x_local_ = torch.cat([x_local, prob_global2local], dim=1)
        prob_local = self.model_local(x_local_)
        prob_local = (prob_local + prob_global2local)/2
        return prob_local, prob_global


