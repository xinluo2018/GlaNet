'''
author: xin luo
create: 2026.1.9  
des: a dual branch U-Net model with attention mechanism
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TransformerFusionBlock import TransformerFusionBlock

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )


class u2net_att_2(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_att_2, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2

        # self.cross_att_4 = BidirectionalCrossAttention(s2_ch=128, dem_ch=128, out_ch=128)
        # self.cross_att_3 = BidirectionalCrossAttention(s2_ch=64, dem_ch=64, out_ch=64)
        # self.cross_att_2 = BidirectionalCrossAttention(s2_ch=32, dem_ch=32, out_ch=32)
        self.cross_att_4 = TransformerFusionBlock(d_model=128, vert_anchors=32, 
                                        horz_anchors=32, h=4, block_exp=4, 
                                        n_layer=1, embd_pdrop=0.1, 
                                        attn_pdrop=0.1, resid_pdrop=0.1)
        self.cross_att_3 = TransformerFusionBlock(d_model=64, vert_anchors=32, 
                                        horz_anchors=32, h=4, block_exp=4, 
                                        n_layer=1, embd_pdrop=0.1, 
                                        attn_pdrop=0.1, resid_pdrop=0.1)
        self.cross_att_2 = TransformerFusionBlock(d_model=32, vert_anchors=32, 
                                        horz_anchors=32, h=4, block_exp=4, 
                                        n_layer=1, embd_pdrop=0.1, 
                                        attn_pdrop=0.1, resid_pdrop=0.1)
        # self.cross_att_1 = TransformerFusionBlock(d_model=32, vert_anchors=32, 
        #                                 horz_anchors=32, h=4, block_exp=4, 
        #                                 n_layer=1, embd_pdrop=0.1, 
        #                                 attn_pdrop=0.1, resid_pdrop=0.1)
        self.conv4 = conv3x3_bn_relu(in_channels=128, out_channels=128)
        self.conv3 = conv3x3_bn_relu(in_channels=64, out_channels=64)
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
        
        self.outp = nn.Sequential(
                        nn.Conv2d(288, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
                        ) 

    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        ## encoder part
        ### branch 1 (scene image)
        x1_b1 = self.down_conv1_b1(x_b1)        # 32        
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: 1/2
        x2_b1 = self.down_conv2_b1(x1_b1)       # 32         
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.down_conv3_b1(x2_b1)       # 64 
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.down_conv4_b1(x3_b1)       # 128  
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16

        ### branch 2 (DEM)
        x1_b2 = self.down_conv1_b2(x_b2)       ##   32     
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: 1/2
        x2_b2 = self.down_conv2_b2(x1_b2)      ##  32     
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x3_b2 = self.down_conv3_b2(x2_b2)      ## 64     
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x4_b2 = self.down_conv4_b2(x3_b2)      ## 128       
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16
        
        ## decoder part
        # level 4
        x4_fuse_att = self.cross_att_4([x4_b1, x4_b2])  # 128
        x4_fuse_att_up = self.up(x4_fuse_att)            #

        # level 3 
        x3_fuse_att = self.cross_att_3([x3_b1, x3_b2])  # 64
        x3_fuse_att = torch.cat([x4_fuse_att_up, x3_fuse_att], dim=1)  # 128+64
        x3_fuse_att_up = self.up(x3_fuse_att)  # 192
        
        # level 2
        x2_fuse_att = self.cross_att_2([x2_b1, x2_b2])  # 32
        x2_fuse_att = torch.cat([x3_fuse_att_up, x2_fuse_att], dim=1)   # 192+32
        x2_fuse_att_up = self.up(x2_fuse_att)  # 224
        # level 1
        # x1_fuse_att = self.cross_att_1([x1_b1, x1_b2])  # 32
        x1_fuse_att = torch.cat([x2_fuse_att_up, x1_b1, x1_b2], dim=1)   # 224+32+32
        x1_fuse_att_up = self.up(x1_fuse_att)            # 288
        # output
        prob = self.outp(x1_fuse_att_up)           # 1

        return prob  

if __name__ == '__main__':
    model = u2net_att_2(num_bands_b1=6, num_bands_b2=1)
    tensor = torch.randn(2, 7, 512, 512)  
    output = model(tensor) 
    print(output.shape) 

