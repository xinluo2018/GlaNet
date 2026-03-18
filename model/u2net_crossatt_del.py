'''
author: xin luo
create: 2025.12.16
des: a dual branch U-Net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        # nn.GroupNorm(num_groups=8, num_channels=out_channels),  
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )
 
class CrossModalChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        # 共享 MLP：输入 2*channels（avg cat max），输出 channels 权重
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )
 
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """
        用 src 的全局统计生成 tgt 的通道注意力权重
        src, tgt: [B, C, H, W]
        return  : [B, C, H, W]  加权后的 tgt
        """
        B, C, _, _ = src.shape
 
        # 对 src 做全局 avg + max 池化，拼接后作为"跨模态通道描述子"
        avg = src.mean(dim=[2, 3])              # [B, C]
        mx = src.amax(dim=[2, 3])
        descriptor = torch.cat([avg, mx], dim=1)  # [B, 2C]
        weight = self.mlp(descriptor)              # [B, C]
        weight = weight.view(B, C, 1, 1)           # [B, C, 1, 1]
        out = tgt * weight + tgt                 # 通道加权 + 残差连接
        return out


class CrossModalSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        # 输入 2ch（avg+max），输出 1ch 空间权重
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8), 
            # nn.GroupNorm(num_groups=8, num_channels=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=kernel_size, padding=pad, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """
        用 src 的空间统计生成 tgt 的空间注意力权重
        src, tgt: [B, C, H, W]
        return  : [B, C, H, W]  加权后的 tgt
        """
        # 对 src 沿通道方向压缩，保留空间维度
        avg = src.mean(dim=1, keepdim=True)      # [B, 1, H, W]
        mx, _ = src.max(dim=1, keepdim=True)     # [B, 1, H, W]
        spatial_desc = torch.cat([avg, mx], dim=1)  # [B, 2, H, W]
 
        weight = self.conv(spatial_desc)            # [B, 1, H, W]
        out = tgt * weight + tgt                     # 空间加权 + 残差连接
        return out
 
class CrossModalAttention(nn.Module):

    def __init__(self,
                 channels: int,
                 reduction: int = 4,
                 spatial_kernel: int = 7):
        super().__init__()
 
        # ── 跨模态通道注意力（双向）──────────────────
        self.cmca_rs2dem = CrossModalChannelAttention(channels, reduction)
        self.cmca_dem2rs = CrossModalChannelAttention(channels, reduction)
 
        # ── 跨模态空间注意力（双向）──────────────────
        self.cmsa_rs2dem = CrossModalSpatialAttention(spatial_kernel)
        self.cmsa_dem2rs = CrossModalSpatialAttention(spatial_kernel)

        # ── 融合投影 ─────────────────────────────────
        # 双向注意力后 cat → 投影回 channels，保持后续通道数不变
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            # nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.ReLU(inplace=True))   # 新增防止过拟合的模块
 
    def forward(self,
                f_rs: torch.Tensor,
                f_dem: torch.Tensor) -> torch.Tensor:
        """
        f_rs : [B, C, H, W]  遥感分支特征
        f_dem: [B, C, H, W]  DEM 分支特征
        return: [B, C, H, W] 跨模态注意力融合后特征
        """
        # ── 方向1：DEM → 引导 RS ─────────────────────
        # Step1: DEM 生成 RS 的空间权重（在通道注意力结果上再施加）
        f_rs_spaatt = self.cmsa_dem2rs(src=f_dem, tgt=f_rs)
        # Step2: DEM 生成 RS 的通道权重
        f_rs_chatt  = self.cmca_dem2rs(src=f_dem, tgt=f_rs)
 
        # ── 方向2：RS → 引导 DEM ──────────────────────
        # Step1: RS 生成 DEM 的空间权重（在通道注意力结果上再施加）
        f_dem_spaatt = self.cmsa_rs2dem(src=f_rs, tgt=f_dem)
        # Step2: RS 生成 DEM 的通道权重
        f_dem_chatt  = self.cmca_rs2dem(src=f_rs, tgt=f_dem)
        # ── 融合 ──────────────────────────────────────
        fused = torch.cat([f_rs_chatt, f_dem_chatt], dim=1)  # [B, 4C, H, W]
        return self.proj(fused)                           # [B, C,  H, W]

class AuxHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
 
    def forward(self, x: torch.Tensor,
                target_size: tuple) -> torch.Tensor:
        x = F.interpolate(x, size=target_size,
                          mode="bilinear", align_corners=False)
        return self.head(x)


class u2net_(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
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
        ### cross-modal attention modules
        self.cma1 = CrossModalAttention(16,  reduction=4)   # 1/2  尺度
        self.cma2 = CrossModalAttention(32,  reduction=4)   # 1/4  尺度
        self.cma3 = CrossModalAttention(64,  reduction=4)   # 1/8  尺度
        self.cma4 = CrossModalAttention(128, reduction=4)   # 1/16 尺度（bottleneck） 

        ## decoder part (fused features)
        self.up_conv1 = conv3x3_bn_relu(192, 64)  # in: 128+64 = 192
        self.up_conv2 = conv3x3_bn_relu(96, 64)   # in: 64+32
        self.up_conv3 = conv3x3_bn_relu(80, 32)   # in: 64+16
 
        # #  深监督辅助头（训练时用，推理时丢弃）
        # self.aux1 = AuxHead(64)   # decoder level1 输出，1/8
        # self.aux2 = AuxHead(64)   # decoder level2 输出，1/4
        # self.aux3 = AuxHead(32)   # decoder level3 输出，1/2
 
        self.outp = nn.Sequential(
                        nn.Conv2d(32, 1, kernel_size=3, padding=1),
                        nn.Sigmoid()
                        ) 

    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        H0, W0 = x.shape[-2:]
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
        ## cross-modal attention
        x1_fused = self.cma1(x1_b1, x1_b2)
        x2_fused = self.cma2(x2_b1, x2_b2)
        x3_fused = self.cma3(x3_b1, x3_b2)
        x4_fused = self.cma4(x4_b1, x4_b2)
        ## decoder part
        x3_skip = torch.cat([self.up(x4_fused), x3_fused], dim=1)  # 128+64 = 192
        x3_decoder = self.up_conv1(x3_skip)
        x2_skip = torch.cat([self.up(x3_decoder), x2_fused], dim=1)   # 64+32 = 96
        x2_decoder = self.up_conv2(x2_skip)       # 96 -> 64
        x1_skip = torch.cat([self.up(x2_decoder), x1_fused], dim=1)      # 64+16 = 80
        x1_decoder = self.up_conv3(x1_skip)          # 80 -> 32
        x1_decoder = self.up(x1_decoder)              # 32 -> 32*2=64 (恢复到输入尺寸的一半)
        prob = self.outp(x1_decoder)           # 1
        # if self.training:
        #     aux1 = self.aux1(x3_decoder, (H0, W0))
        #     aux2 = self.aux2(x2_decoder, (H0, W0))
        #     aux3 = self.aux3(x1_decoder, (H0, W0))
        #     return prob, aux1, aux2, aux3
        return prob


if __name__ == "__main__":
    model = u2net_(num_bands_b1=6, num_bands_b2=1)
    model.eval()
    x = torch.randn(2, 7, 256, 256)  # batch_size=2, num_bands=7, H=256, W=256
    prob = model(x)
    print(prob.shape)  # 应输出: torch.Size([2, 1, 256, 256])
    model.train()
    prob, aux1, aux2, aux3 = model(x)
    print(prob.shape, aux1.shape, aux2.shape, aux3.shape)  

