'''
author: xin luo
create: 2026.1.18
des: a dual branch U-Net model with DEM adjustment
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class LatitudinalCorrection(nn.Module):
    def __init__(self, k=0.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k))
    def forward(self, dem, latitude):
        latitude = latitude.view(-1, 1, 1, 1)
        lat_adj = (torch.abs(latitude)-45)/45  ## range: -1 to 1 
        # non-linear scaling
        dem_adj = dem + self.k * lat_adj  # adjust DEM values
        return dem_adj


class u2net_dem_adj(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_dem_adj, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1: scene image
        self.down_conv1_b1 = conv3x3_bn_relu(self.num_bands_b1, 16)        
        self.down_conv2_b1 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b1 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b1 = conv3x3_bn_relu(64, 128)
        ### branch 2: DEM
        self.lat_correction = LatitudinalCorrection()
        self.down_conv1_b2 = conv3x3_bn_relu(self.num_bands_b2, 16)
        self.down_conv2_b2 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b2 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b2 = conv3x3_bn_relu(64, 128)
        ## decoder part (fused features)
        self.up_conv1 = conv3x3_bn_relu(384, 64)  # in: 128+128+64+64 = 384
        self.up_conv2 = conv3x3_bn_relu(128, 64)   # in: 64+32+32
        self.up_conv3 = conv3x3_bn_relu(96, 32)   # in: 64+16+16        
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
                        ) 
    def forward(self, x, latitude):       ## input size: 7x256x256
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
        x1_b2_adjust = self.lat_correction(x1_b2, latitude=latitude) # adjust DEM with latitude info
        x2_b2 = self.down_conv2_b2(x1_b2)    
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x2_b2_adjust = self.lat_correction(x2_b2, latitude=latitude) # adjust DEM with latitude info
        x3_b2 = self.down_conv3_b2(x2_b2) 
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x3_b2_adjust = self.lat_correction(x3_b2, latitude=latitude) # adjust DEM with latitude info
        x4_b2 = self.down_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16
        x4_b2_adjust = self.lat_correction(x4_b2, latitude=latitude) # adjust DEM with latitude info
        ## decoder part
        x4_up = torch.cat([self.up(x4_b1), self.up(x4_b2_adjust), 
                                    x3_b1, x3_b2_adjust], dim=1)  # 128+128+64+64 # type: ignore 
        x3_up = self.up_conv1(x4_up)        
        x3_up = torch.cat([self.up(x3_up), x2_b1, x2_b2_adjust], dim=1)   # 64+32+32 # type: ignore 
        x2_up = self.up_conv2(x3_up)       
        x2_up = torch.cat([self.up(x2_up), x1_b1, x1_b2_adjust], dim=1)   # 64+16+16 # type: ignore 
        x1_up = self.up_conv3(x2_up)        #  32
        x1_up = self.up(x1_up)              #  
        prob = self.outp(x1_up)             # 1
        return prob          

if __name__ == "__main__":
    model = u2net_dem_adj(num_bands_b1=6, num_bands_b2=1)
    # batch size of 2, 7 channels (6 for scene image + 1 for DEM), 512x512
    input_tensor = torch.randn(2, 7, 512, 512)  
    latitude = torch.randn(2, 1)  # batch size of 2, 1 latitude value per sample
    output = model(input_tensor, latitude)
    print(output.shape)  # should be [2, 1, 256, 256]
