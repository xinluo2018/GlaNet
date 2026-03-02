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
        output=self.sigmoid(max_out+avg_out) #
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

class CrossAttentionFusion(nn.Module):
    def __init__(self, s2_ch, dem_ch, out_ch):
        super().__init__()
        self.query = nn.Conv2d(s2_ch, out_ch, kernel_size=1)
        self.key   = nn.Conv2d(dem_ch, out_ch, kernel_size=1)
        self.value = nn.Conv2d(dem_ch, out_ch, kernel_size=1)        
        self.softmax = nn.Softmax(dim=-1)
        self.scale = out_ch ** -0.5   

    def forward(self, x_s2, x_dem):
        b, c, h, w = x_s2.size()
        
        # 1. 生成 Q, K, V 并展平
        q = self.query(x_s2).view(b, -1, h * w)  # [B, C, N]
        k = self.key(x_dem).view(b, -1, h * w).permute(0, 2, 1)  # [B, N, C]
        v = self.value(x_dem).view(b, -1, h * w)  # [B, C, N]
        
        # 2. 计算交叉注意力图 (S2 vs DEM)
        # 这一步在问：S2的光谱像素与DEM的哪些地形位置最匹配？
        attn = self.softmax(torch.bmm(k, q) * self.scale)  # [B, N, N]
        
        # 3. 用注意力图加权 DEM 的 Value
        out = torch.bmm(v, attn).view(b, -1, h, w)
        
        # 4. 与原始 S2 特征融合
        return out + x_s2

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, s2_ch, dem_ch, out_ch):
        super().__init__()
        # use s2 as a query attention 
        self.attn_s2_query = CrossAttentionFusion(s2_ch, dem_ch, out_ch)
        # use dem as a query attention
        self.attn_dem_query = CrossAttentionFusion(dem_ch, s2_ch, out_ch)
        
        # define LayerNorm layers for both outputs
        self.ln_s2 = nn.LayerNorm(out_ch)
        self.ln_dem = nn.LayerNorm(out_ch)
        
        ## fuse the two attention results
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1)
        )

    def forward(self, x_s2, x_dem):
        # 1. calculate bidirectional cross attention
        out_s2 = self.attn_s2_query(x_s2, x_dem)
        out_dem = self.attn_dem_query(x_dem, x_s2)
        
        # 2. apply LayerNorm
        # LayerNorm requires [B, N, C] format, so we need to reshape
        b, c, h, w = out_s2.size()
        
        # apply LN to out_s2
        out_s2 = out_s2.view(b, c, -1).permute(0, 2, 1) # [B, N, C]
        out_s2 = self.ln_s2(out_s2)
        out_s2 = out_s2.permute(0, 2, 1).view(b, c, h, w) # restore [B, C, H, W]
        
        # 对 out_dem 进行 LN
        out_dem = out_dem.view(b, c, -1).permute(0, 2, 1) # [B, N, C]
        out_dem = self.ln_dem(out_dem)
        out_dem = out_dem.permute(0, 2, 1).view(b, c, h, w) # 还原 [B, C, H, W]
        
        # 3. 拼接并融合
        combined = torch.cat([out_s2, out_dem], dim=1)
        return self.fusion(combined)


class u2net_att_1(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_att_1, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        # # self.att_1_b1 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_2 = CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.att_3 = CBAMBlock(channel=128, reduction=4, kernel_size=7)
        self.att_4 = CBAMBlock(channel=256, reduction=4, kernel_size=7)


        # self.cross_att_4 = CrossAttentionFusion(s2_ch=128, dem_ch=128, out_ch=128)
        # self.cross_att_3 = CrossAttentionFusion(s2_ch=64, dem_ch=64, out_ch=64)
        # self.cross_att_2 = CrossAttentionFusion(s2_ch=32, dem_ch=32, out_ch=32)
        # self.cross_att_1 = CrossAttentionFusion(s2_ch=32, dem_ch=32, out_ch=32)
        self.cross_att_4 = BidirectionalCrossAttention(s2_ch=128, dem_ch=128, out_ch=128)
        self.cross_att_3 = BidirectionalCrossAttention(s2_ch=64, dem_ch=64, out_ch=64)
        self.cross_att_2 = BidirectionalCrossAttention(s2_ch=32, dem_ch=32, out_ch=32)

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
        self.up_conv4 = conv3x3_bn_relu(128, 128)   # 
        self.up_conv3 = conv3x3_bn_relu(192, 128)   # in:
        self.up_conv2 = conv3x3_bn_relu(192, 128)  # in: 
        self.up_conv1 = conv3x3_bn_relu(192, 64)  # in: 

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
        x1_b1 = self.down_conv1_b1(x_b1)        # 32        
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: 1/2
        x2_b1 = self.down_conv2_b1(x1_b1)       # 32         
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.down_conv3_b1(x2_b1)       # 64 
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.down_conv4_b1(x3_b1)       # 128  
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        # x4_att_b1 = self.att_4_b1(x4_b1)  # 128 

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
        x4_fuse_att = self.cross_att_4(x4_b1, x4_b2)  # 128
        # x4_fuse = torch.cat([x4_b1, x4_b2], dim=1)  # 256
        # x4_fuse_att = self.att_4(x4_fuse)  # 256
        x4_fuse_att = self.up_conv4(x4_fuse_att)     # 128
        # level 3 
        # x3_att_b1 = self.att_3_b1(x3_b1)
        # x3_att_b2 = self.att_3_b2(x3_b2)         
        x3_fuse_att = self.cross_att_3(x3_b1, x3_b2)  # 64
        # x3_fuse = torch.cat([x3_b1, x3_b2], dim=1)  # 128
        # x3_fuse_att = self.att_3(x3_fuse)  #
        x3_att_fuse = torch.cat([self.up(x4_fuse_att), x3_fuse_att], dim=1)  # 128+64
        x3_att_fuse = self.up_conv3(x3_att_fuse)     # 128   

        # level 2
        # x2_fuse_att = self.cross_att_2(x2_b1, x2_b2)  # 32
        # x2_att_b1 = self.att_2_b1(x2_b1)
        # x2_att_b2 = self.att_2_b2(x2_b2)         
        # x2_fuse = torch.cat([x2_b1, x2_b2], dim=1)  # 64
        # x2_fuse_att = self.att_2(x2_fuse)  #
        x2_att_fuse = torch.cat([self.up(x3_att_fuse), x2_b1, x2_b2], dim=1)   # 128+32+32
        x2_att_fuse = self.up_conv2(x2_att_fuse)    # 128

        # level 1
        # x1_fuse_att = self.cross_att_1(x1_b1, x1_b2)  # 32
        x1 = torch.cat([self.up(x2_att_fuse), x1_b1, x1_b2], dim=1)   # 128+32+32
        x1 = self.up_conv1(x1)  ##  64

        x1 = self.up(x1)            #
        prob = self.outp(x1)           # 1
        return prob          


if __name__ == '__main__':
    model = u2net_att_1(num_bands_b1=6, num_bands_b2=1)
    tensor = torch.randn(2, 7, 512, 512)  
    output = model(tensor) 
    print(output.shape) 

