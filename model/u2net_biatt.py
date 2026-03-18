'''
author: xin luo
create: 2026.1.9  
des: a dual branch U2Net model with bidirectional attention mechanism
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out

class AdaptivePool2d(nn.Module):
    ## output_h and output_w are the target output size after pooling
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x
        return y

class CrossAttentionFusion(nn.Module):
    ''' 
    d_opt_fea: number of channels for optical features (query)
    d_dem_fea: number of channels for DEM features (key and value)
    d_model: number of channels for the output fused features
    head: number of attention heads (for multi-head attention, not implemented in this version)
    '''
    def __init__(self, d_opt_fea, d_dem_fea, d_model, head=1):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_model // head
        self.d_value = d_model // head
        self.head = head
        self.query = nn.Linear(d_opt_fea, head * self.d_key)
        self.key   = nn.Linear(d_dem_fea, head * self.d_key)
        self.value = nn.Linear(d_dem_fea, head * self.d_value)
        self.out_proj = nn.Linear(head * self.d_value, d_model)  # output projection
        self.softmax = nn.Softmax(dim=-1)
        self.ln = nn.LayerNorm(d_model)


    def forward(self, x_1, x_2):  
        b_s, n_l = x_1.shape[0], x_1.shape[1]
        q = self.query(x_1).view(b_s, n_l, self.head, self.d_key).permute(0, 2, 1, 3)  # [B, head, N, d_k]
        k = self.key(x_2).view(b_s, n_l, self.head, self.d_key).permute(0, 2, 3, 1) # [B, head, d_k, N]
        v = self.value(x_2).view(b_s, n_l, self.head, self.d_value).permute(0, 2, 1, 3)  # [B, head, N, d_v]
        
        # 2. 计算交叉注意力图 (x_1 vs x_2)
        # 这一步在问：x_1的特征与x_2的哪些特征最匹配？
        attn = torch.matmul(q, k) / np.sqrt(self.d_key)  # [B, head, N, N] 注意力图，表示每个x_1特征与x_2所有特征的相关性
        attn = self.softmax(attn) 
        # 3. 用注意力图加权 x_2 的 Value
        out = torch.matmul(attn, v)   # [B, head, N, d_v]
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, n_l, -1)  # [B, N, C]
        out = self.out_proj(out)   # output projection
        # 4. skip connection
        out = out + x_1  # [B, N, C]
        out = self.ln(out)   # LayerNorm
        return out

class BidirectionalCrossAttention(nn.Module):
    ## vert_anchors and horz_anchors are the target spatial dimensions for attention computation
    def __init__(self, d_opt_fea, d_dem_fea, d_model, 
                            vert_anchors=16, horz_anchors=16, head=4):        
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.head = head

        # # positional embedding parameter (learnable), opt_fea + dem_fea
        # self.pos_emb_opt = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        # self.pos_emb_dem = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        # spatial dimension reduction for attention        
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.opt_coefficient = LearnableWeights()
        self.dem_coefficient = LearnableWeights()

        # use opt as a query attention 
        self.attn_opt_query = CrossAttentionFusion(d_opt_fea, d_dem_fea, d_model, head=head)
        # use dem as a query attention
        self.attn_dem_query = CrossAttentionFusion(d_dem_fea, d_opt_fea, d_model, head=head)
                
        ## fuse the two attention results
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model), 
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

    def forward(self, x_opt, x_dem):
        ## reduced features for attention
        ## optical image
        opt_fea_ = self.opt_coefficient(self.avgpool(x_opt), self.maxpool(x_opt)) 
        # opt_fea_ = self.avgpool(x_opt) + self.maxpool(x_opt)  # [B, C, vert_anchors, horz_anchors]
        bs, new_c, new_h, new_w = opt_fea_.shape[0], opt_fea_.shape[1], opt_fea_.shape[2], opt_fea_.shape[3]
        opt_fea_flat = opt_fea_.contiguous().view(bs, new_c, -1).permute(0, 2, 1)  # [B, N, C]
        ## dem 
        dem_fea_ = self.dem_coefficient(self.avgpool(x_dem), self.maxpool(x_dem))
        # dem_fea_ = self.avgpool(x_dem) + self.maxpool(x_dem)  # [B, C, vert_anchors, horz_anchors]
        dem_fea_flat = dem_fea_.contiguous().view(bs, new_c, -1).permute(0, 2, 1)  # [B, N, C]
        # 1. calculate bidirectional cross attention
        opt_fea = self.attn_opt_query(x_1 = opt_fea_flat, x_2 = dem_fea_flat) # [B, N, C]
        opt_fea = opt_fea.permute(0, 2, 1).contiguous().view(bs, -1, new_h, new_w)  # restore [B, C, H, W]
        dem_fea = self.attn_dem_query(x_1 = dem_fea_flat, x_2 = opt_fea_flat) # [B, N, C]
        dem_fea = dem_fea.permute(0, 2, 1).contiguous().view(bs, -1, new_h, new_w)  # restore [B, C, H, W]
        # 2. restore spatial dimensions for both features (using interpolation)
        opt_fea_out = F.interpolate(opt_fea, size=x_opt.shape[2:], mode='bilinear')  # [B, C, H, W]
        dem_fea_out = F.interpolate(dem_fea, size=x_dem.shape[2:], mode='bilinear')  # [B, C, H, W]
        # 3. 拼接并融合, with skip connection
        combined = torch.cat([opt_fea_out, dem_fea_out], dim=1)
        return self.fusion(combined)

class u2net_biatt(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_biatt, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        ## when setting head>1, performance does not improve
        self.cross_att_4 = BidirectionalCrossAttention(d_opt_fea=128, d_dem_fea=128, 
                                        d_model=128, vert_anchors=16, horz_anchors=16, head=1)
        self.cross_att_3 = BidirectionalCrossAttention(d_opt_fea=64, d_dem_fea=64, 
                                        d_model=64, vert_anchors=16, horz_anchors=16, head=1)
        self.cross_att_2 = BidirectionalCrossAttention(d_opt_fea=32, d_dem_fea=32, 
                                        d_model=32, vert_anchors=16, horz_anchors=16, head=1)
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
        self.up_conv4 = conv3x3_bn_relu(128, 128)  # 
        self.up_conv3 = conv3x3_bn_relu(192, 128)  # 
        self.up_conv2 = conv3x3_bn_relu(192, 128)  #  
        self.up_conv1 = conv3x3_bn_relu(192, 64)   # 

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
        x4_fuse_att = self.up_conv4(x4_fuse_att)     # 128
        # level 3 
        x3_fuse_att = self.cross_att_3(x3_b1, x3_b2)  # 64
        x3_att_fuse = torch.cat([self.up(x4_fuse_att), x3_fuse_att], dim=1)  # 128+64
        x3_att_fuse = self.up_conv3(x3_att_fuse)     # 128   

        # level 2
        x2_att_fuse = torch.cat([self.up(x3_att_fuse), x2_b1, x2_b2], dim=1)   # 128+32+32
        x2_att_fuse = self.up_conv2(x2_att_fuse)    # 128

        # level 1
        x1 = torch.cat([self.up(x2_att_fuse), x1_b1, x1_b2], dim=1)   # 128+32+32
        x1 = self.up_conv1(x1)  ##  64

        x1 = self.up(x1)            #
        prob = self.outp(x1)           # 1
        return prob          


if __name__ == '__main__':
    model = u2net_biatt(num_bands_b1=6, num_bands_b2=1)
    tensor = torch.randn(2, 7, 512, 512)  
    output = model(tensor) 
    print(output.shape) 

