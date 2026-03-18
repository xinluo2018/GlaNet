'''
author: xin luo
create: 2025.12.16
des: a dual branch U-Net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
try:    
    from .attention import CBAMBlock
except ImportError:
    from attention import CBAMBlock


def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class u2net_cbam(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_cbam, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.down_conv1_b1 = conv3x3_bn_relu(self.num_bands_b1, 16)        
        self.cbam1 = CBAMBlock(channel=16, reduction=4, kernel_size=7)  # CBAM attention after 1st block
        self.down_conv2_b1 = conv3x3_bn_relu(16, 32)
        self.cbam2 = CBAMBlock(channel=32, reduction=4, kernel_size=7)  # CBAM attention after 2nd block
        self.down_conv3_b1 = conv3x3_bn_relu(32, 64)
        self.cbam3 = CBAMBlock(channel=64, reduction=4, kernel_size=7)  # CBAM attention after 3rd block
        self.down_conv4_b1 = conv3x3_bn_relu(64, 128)
        self.cbam4 = CBAMBlock(channel=128, reduction=4, kernel_size=7)  # CBAM attention after 4th block
        ### branch 2
        self.down_conv1_b2 = conv3x3_bn_relu(self.num_bands_b2, 16)
        self.down_conv2_b2 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b2 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b2 = conv3x3_bn_relu(64, 128)
        ## decoder part (fused features)
        self.up_conv1 = conv3x3_bn_relu(384, 64)  # in: 128+128+64+64
        self.up_conv2 = conv3x3_bn_relu(128, 64)   # in: 64+32+32
        self.up_conv3 = conv3x3_bn_relu(96, 32)   # in: 64+16+16
        
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
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
        x1_b1 = self.cbam1(x1_b1)                         # CBAM attention
        x2_b1 = self.down_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x2_b1 = self.cbam2(x2_b1)                         # CBAM attention
        x3_b1 = self.down_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x3_b1 = self.cbam3(x3_b1)                         # CBAM attention
        x4_b1 = self.down_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        x4_b1 = self.cbam4(x4_b1)                         # CBAM attention
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
        x4_up = torch.cat([self.up(x4_b1), self.up(x4_b2), x3_b1, x3_b2], dim=1)  # 128+128+64+64
        x3_up = self.up_conv1(x4_up)        
        x3_up = torch.cat([self.up(x3_up), x2_b1, x2_b2], dim=1)   # 64+32+32
        x2_up = self.up_conv2(x3_up)       
        x2_up = torch.cat([self.up(x2_up), x1_b1, x1_b2], dim=1)   # 64+16+16
        x1_up = self.up_conv3(x2_up)        #  32
        x1_up = self.up(x1_up)              #  
        prob = self.outp(x1_up)           # 1
        return prob          

if __name__ == "__main__":
    model = u2net_cbam(num_bands_b1=3, num_bands_b2=4)
    input_tensor = torch.randn(1, 7, 256, 256)  # batch_size=1, total_channels=7 (3 for branch 1 and 4 for branch 2), height=256, width=256
    output = model(input_tensor)
    print(output.shape)  # should be [1, 1, 256, 256] 

