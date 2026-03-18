'''
author: xin luo
create: 2026.3.6
des: attention modules
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        # Sobel卷积核（水平和垂直方向）
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                    stride=1, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                    stride=1, padding=1, groups=in_channels, bias=False) 

        # 固定Sobel卷积核参数
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().repeat(in_channels, 1, 1, 1)
        
        self.sobel_x.weight.data = sobel_kernel_x  # 将Sobel卷积核赋值给卷积层权重
        self.sobel_y.weight.data = sobel_kernel_y
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        # 注意力生成层
        self.conv_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算梯度特征
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)  # 梯度幅值
        
        # 生成注意力图
        att = self.conv_att(edge)
        return x * att + x  # 残差连接
