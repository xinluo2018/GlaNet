'''
author: xin luo
create: 2026.1.9  
des: a dual branch U-Net model with attention mechanism
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel, 1, bias=True)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2, 1, 
                            kernel_size = kernel_size, 
                            padding = kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):
    def __init__(self, channel=256, reduction=4, kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual    ## residual connection

class u2net_att_1(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_att_1, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.att_1_b1 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_2_b1 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_3_b1 = CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.att_4_b1 = CBAMBlock(channel=128, reduction=4, kernel_size=7)

        self.att_1_b2 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_2_b2 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_3_b2 = CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.att_4_b2 = CBAMBlock(channel=128, reduction=4, kernel_size=7)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.down_conv1_b1 = conv3x3_bn_relu(self.num_bands_b1, 32)        
        self.down_conv2_b1 = conv3x3_bn_relu(32, 32)
        self.down_conv3_b1 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b1 = conv3x3_bn_relu(64, 128)
        ### branch 2
        self.down_conv1_b2 = conv3x3_bn_relu(self.num_bands_b2, 32)
        self.down_conv2_b2 = conv3x3_bn_relu(32, 32)
        self.down_conv3_b2 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b2 = conv3x3_bn_relu(64, 128)
        ## decoder part (fused features)
        self.up_conv1 = conv3x3_bn_relu(384, 64)  # in: 128+128+128 = 384
        self.up_conv2 = conv3x3_bn_relu(128, 64)   # in: 64+32+32
        self.up_conv3 = conv3x3_bn_relu(128, 64)   # in: 32+32+32
        
        self.outp = nn.Sequential(
                        nn.Conv2d(64, 1, kernel_size=3, padding=1),
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
        x2_b1 = self.down_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.down_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.down_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        x4_att_b1 = self.att_4_b1(x4_b1)  # 128 

        ### branch 2 (DEM)
        x1_b2 = self.down_conv1_b2(x_b2)              
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: 1/2
        x2_b2 = self.down_conv2_b2(x1_b2)              
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x3_b2 = self.down_conv3_b2(x2_b2)              
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x4_b2 = self.down_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16
        x4_att_b2 = self.att_4_b2(x4_b2)   ## 128
        
        ## decoder part
        x3_att_b1 = self.att_3_b1(x3_b1)  # 64 
        x3_att_b2 = self.att_3_b2(x3_b2)  # 64 
        x3_up = torch.cat([self.up(x4_att_b1), self.up(x4_att_b2), 
                           x3_att_b1, x3_att_b2], dim=1)  # 128+128+64+64
        x3_up = self.up_conv1(x3_up)     # 64    

        x2_att_b1 = self.att_2_b1(x2_b1)  # 32
        x2_att_b2 = self.att_2_b2(x2_b2)  # 32
        x2_up = torch.cat([self.up(x3_up), x2_att_b1, x2_att_b2], dim=1)   # 64+32+32
        x2_up = self.up_conv2(x2_up)    # 64

        x1_att_b1 = self.att_1_b1(x1_b1)  # 32
        x1_att_b2 = self.att_1_b2(x1_b2)  # 32
        x1_up = torch.cat([self.up(x2_up), x1_att_b1, x1_att_b2], dim=1)   # 64+32+32
        x1_up = self.up_conv3(x1_up)

        x1_up = self.up(x1_up)              #
        prob = self.outp(x1_up)           # 1

        return prob          


if __name__ == '__main__':
    model = u2net_att_1(num_bands_b1=6, num_bands_b2=1)
    tensor = torch.randn(2, 7, 512, 512)  
    output = model(tensor) 
    print(output.shape) 

