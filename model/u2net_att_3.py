'''
author: xin luo
create: 2026.1.9  
des: a dual branch U-Net model with attention mechanism (fixed decoder fusion)
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
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual    ## residual connection

class FusionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 学习三个输入的空间权重（按通道共享）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, max(4, in_channels // 2), 1),
            nn.ReLU(),
            nn.Conv2d(max(4, in_channels // 2), 3, 1), # 输出三个权重的通道
            nn.Softmax(dim=1)
        )

    def forward(self, b1, b2, b3):
        # b1/b2/b3: (B,C,H,W) — 通道数应相同 for each input
        cat_feat = torch.cat([b1, b2, b3], dim=1)
        weights = self.conv(cat_feat)  # (B,3,H,W)
        out = b1 * weights[:, 0:1, :, :] + b2 * weights[:, 1:2, :, :] + b3 * weights[:, 2:3, :, :]
        return out

class u2net_att_3(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_att_3, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.att_1 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_2 = CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.att_3 = CBAMBlock(channel=128, reduction=4, kernel_size=7)
        self.att_4_b1 = CBAMBlock(channel=128, reduction=4, kernel_size=7)
        self.att_4_b2 = CBAMBlock(channel=128, reduction=4, kernel_size=7)

        self.fusion_att_1 = FusionAttention(in_channels=32+32+32)   # 32*3
        self.fusion_att_2 = FusionAttention(in_channels=64+64+64)    # 64*3
        self.fusion_att_3 = FusionAttention(in_channels=128+128+128) # 128*3
        
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.down_conv1_b1 = conv3x3_bn_relu(self.num_bands_b1, 16)        
        self.down_conv2_b1 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b1 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b1 = conv3x3_bn_relu(64, 128)
        ### branch 2
        self.down_conv1_b2 = conv3x3_bn_relu(self.num_bands_b2, 16)
        self.down_conv2_b2 = conv3x3_bn_relu(16, 32)
        self.down_conv3_b2 = conv3x3_bn_relu(32, 64)
        self.down_conv4_b2 = conv3x3_bn_relu(64, 128)
        ## decoder part (adjusted channel sizes to match fusion outputs)
        self.up_conv1 = conv3x3_bn_relu(128, 64)   # fused level3 -> reduce to 64
        self.up_conv2 = conv3x3_bn_relu(128, 64)   # concat 64+64 -> 128 -> reduce to 64
        self.up_conv3 = conv3x3_bn_relu(96, 32)    # concat 64+32 -> 96 -> reduce to 32
        
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
                        ) 

    def forward(self, x):       ## input size: (B, num_bands_b1+num_bands_b2, H, W)
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        ## encoder part
        ### branch 1 (scene image)
        x1_b1 = self.down_conv1_b1(x_b1)                 
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: H/2
        x2_b1 = self.down_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: H/4
        x3_b1 = self.down_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: H/8
        x4_b1 = self.down_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: H/16
        x4_b1 = self.att_4_b1(x4_b1)

        ### branch 2 (DEM)
        x1_b2 = self.down_conv1_b2(x_b2)              
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: H/2
        x2_b2 = self.down_conv2_b2(x1_b2)              
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: H/4 
        x3_b2 = self.down_conv3_b2(x2_b2)              
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: H/8
        x4_b2 = self.down_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: H/16
        x4_b2 = self.att_4_b2(x4_b2)
        ## apply attention on concatenated per-level features (channel dims match att blocks)
        # level3 (deep-middle): concat x3_b1 (64) + x3_b2 (64) -> 128
        x3_cat = torch.cat([x3_b1, x3_b2], dim=1)
        x3_att = self.att_3(x3_cat)  # 128

        # fuse level4-up -> level3 context using: up(x4_b1), up(x4_b2), x3_att
        up4_b1 = self.up(x4_b1)  # H/8, C=128
        up4_b2 = self.up(x4_b2)  # H/8, C=128
        fused3 = self.fusion_att_3(up4_b1, up4_b2, x3_att)  # output C=128
        x3_up = self.up_conv1(fused3)   # -> 64 ch

        # level2 attention: concat x2_b1 (32) + x2_b2 (32) -> 64
        x2_cat = torch.cat([x2_b1, x2_b2], dim=1)
        x2_att = self.att_2(x2_cat)  # 64

        # produce fused level2 from up(x3_b1), up(x3_b2), x2_att
        up3_b1 = self.up(x3_b1)  # H/4, C=64
        up3_b2 = self.up(x3_b2)  # H/4, C=64
        fused2 = self.fusion_att_2(up3_b1, up3_b2, x2_att)  # C=64

        # combine decoder feature (upsampled x3_up) with fused2
        x3_up_us = self.up(x3_up)  # H/4, C=64
        x2_cat_dec = torch.cat([x3_up_us, fused2], dim=1)  # 64+64=128
        x2_up = self.up_conv2(x2_cat_dec)  # -> 64

        # level1 attention: concat x1_b1 (16) + x1_b2 (16) -> 32
        x1_cat = torch.cat([x1_b1, x1_b2], dim=1)
        x1_att = self.att_1(x1_cat)  # 32

        # fused level1 from up(x2_b1), up(x2_b2), x1_att
        up2_b1 = self.up(x2_b1)  # H/2, C=32
        up2_b2 = self.up(x2_b2)  # H/2, C=32
        fused1 = self.fusion_att_1(up2_b1, up2_b2, x1_att)  # C=32

        # combine decoder feature with fused1
        x2_up_us = self.up(x2_up)  # H/2, C=64
        x1_cat_dec = torch.cat([x2_up_us, fused1], dim=1)  # 64+32=96
        x1_up = self.up_conv3(x1_cat_dec)  # -> 32

        x1_up = self.up(x1_up)              # H, 32
        prob = self.outp(x1_up)           # 1

        return prob