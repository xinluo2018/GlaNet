'''
author: xin luo
create: 2026.3.15
des: a dual branch U-Net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# class GlobalBatchNorm2d(nn.Module):
#     """
#     One mean/var for the whole mini-batch over (N,C,H,W).
#     Optional affine: per-channel gamma/beta (or you can make them scalar).
#     """
#     def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
#         super().__init__()
#         self.eps = eps
#         self.affine = affine
#         if affine:
#             self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
#             self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)
#     def forward(self, x):
#         # x: (N,C,H,W)
#         mean = x.mean(dim=(0,1,2,3), keepdim=True)
#         var  = x.var(dim=(0,1,2,3), keepdim=True, unbiased=False)
#         x_hat = (x - mean) / torch.sqrt(var + self.eps)
#         if self.affine:
#             x_hat = x_hat * self.weight + self.bias
#         return x_hat

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        # nn.BatchNorm2d(out_channels),
        # nn.GroupNorm(1, out_channels),  # group normalization  
        # GlobalBatchNorm2d(out_channels),  # global batch normalization      
        nn.ReLU(inplace=True)
        )


class u2net(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.down_conv1_b1 = conv(self.num_bands_b1, 16)        
        self.down_conv2_b1 = conv(16, 32)
        self.down_conv3_b1 = conv(32, 64)
        self.down_conv4_b1 = conv(64, 128)
        ### branch 2
        self.down_conv1_b2 = conv(self.num_bands_b2, 16)  # for DEM, no batch norm
        self.down_conv2_b2 = conv(16, 32)
        self.down_conv3_b2 = conv(32, 64)
        self.down_conv4_b2 = conv(64, 128)
        ## decoder part (fused features)
        ## fusion 
        self.decoder4 = conv(256, 64)   # in: 
        self.decoder3 = conv(192, 64)   # in: 
        self.decoder2 = conv(128, 64)   # in: up(64) + fused(64)
        self.decoder1 = conv(96, 32)   # in: up(64) + fused(32)
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        ) 

    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        ## encoder part
        ### branch 1 (scene image)
        x1_b1 = self.down_conv1_b1(x_b1)                 
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: 1/2
        x2_b1 = self.down_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.down_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.down_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        ### branch 2 (DEM)
        x1_b2 = self.down_conv1_b2(x_b2)              
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: 1/2
        x2_b2 = self.down_conv2_b2(x1_b2)              
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x3_b2 = self.down_conv3_b2(x2_b2)              
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x4_b2 = self.down_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16
        ## decoder part
        x4_fus = torch.cat([x4_b1, x4_b2], dim=1)  # 128+128
        x4_decoder = self.decoder4(x4_fus)   ## 
        x3_fus = torch.cat([x3_b1, x3_b2], dim=1)  # 64+64
        x3_decoder = torch.cat([self.up(x4_decoder), x3_fus], dim=1)  # 128+64 
        x3_decoder = self.decoder3(x3_decoder)
        x2_fus = torch.cat([x2_b1, x2_b2], dim=1)  #
        x2_decoder = torch.cat([self.up(x3_decoder), x2_fus], dim=1)   # 64+64
        x2_decoder = self.decoder2(x2_decoder)
        x1_fus = torch.cat([x1_b1, x1_b2], dim=1)  #
        x1_decoder = torch.cat([self.up(x2_decoder), x1_fus], dim=1)   # 64+32
        x1_decoder = self.decoder1(x1_decoder)
        x1_decoder = self.up(x1_decoder)  # up to original size
        logit = self.outp(x1_decoder)  # 1x256x256
        return logit

if __name__ == '__main__':
    model = u2net(num_bands_b1=6, num_bands_b2=1)
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=W=256
    out = model(x)
    print(out.shape)  # should be [2, 1, 256, 256] 


