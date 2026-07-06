'''
author: xin luo
create: 2026.5.5
des: 
(1) a tripple branch U-Net model with Swin Transformer fusion, better than unet
(2) cbam attention improve the performance
todo: 
(1) cross vitransformer ~ cross swintransformer
'''

import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath  # type: ignore

def conv(in_channels, out_channels, 
                kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        # nn.GroupNorm(1, out_channels),  # group normalization  
        # GlobalBatchNorm2d(out_channels),  # global batch normalization      
        nn.ReLU(inplace=True)
        )

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 8)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.shared_mlp=nn.Sequential(
            nn.Conv2d(channel, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1, bias=True),
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.shared_mlp(max_result)
        avg_out=self.shared_mlp(avg_result)
        att = self.sigmoid(max_out+avg_out)
        return x*att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2, 1, kernel_size = kernel_size, 
                            padding = kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        att = self.sigmoid(output)
        return  x*att

class CBAMBlock(nn.Module):
    def __init__(self, channel=256, reduction=4, kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=self.ca(x)
        out=self.sa(out)
        return out+residual    ## residual connection

class ssf_fusion_xyd(nn.Module):
    """ cross channel attention fusion for x,y,d features """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden = max(in_channels // reduction, 16)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        def make_mlp():
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, in_channels, 1, bias=True))

        self.mlp_xy = make_mlp(); self.mlp_xd = make_mlp()
        self.mlp_yx = make_mlp(); self.mlp_yd = make_mlp()
        self.mlp_dx = make_mlp(); self.mlp_dy = make_mlp()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, d):
        # x: fusing y、d
        ch_xy = self.sigmoid(self.mlp_xy(self.maxpool(y)) + self.mlp_xy(self.avgpool(y)))
        ch_xd = self.sigmoid(self.mlp_xd(self.maxpool(d)) + self.mlp_xd(self.avgpool(d)))
        x_out = x + x * ch_xy + x * ch_xd

        # y: fusing x、d
        ch_yx = self.sigmoid(self.mlp_yx(self.maxpool(x)) + self.mlp_yx(self.avgpool(x)))
        ch_yd = self.sigmoid(self.mlp_yd(self.maxpool(d)) + self.mlp_yd(self.avgpool(d)))
        y_out = y + y * ch_yx + y * ch_yd

        # d: fusing x、y
        ch_dx = self.sigmoid(self.mlp_dx(self.maxpool(x)) + self.mlp_dx(self.avgpool(x)))
        ch_dy = self.sigmoid(self.mlp_dy(self.maxpool(y)) + self.mlp_dy(self.avgpool(y)))
        d_out = d + d * ch_dx + d * ch_dy
        return x_out, y_out, d_out


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """ Partition into non-overlapping windows """
    B, H, W, C = x.shape        
    # If feature map is smaller than window size, pad it
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)        
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) 
    return windows  ## [num_windows*B, window_size, window_size, C]

def window_reverse(windows: torch.Tensor, 
                    window_size: int, H: int, W: int) -> torch.Tensor:
    """ Reverse window partition """
    # Calculate number of windows
    num_windows_h = H // window_size   # upper bound for number of windows in height
    num_windows_w = W // window_size   
    B = windows.shape[0] // (num_windows_h * num_windows_w)        
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)        
    return x

class Attention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """
    def __init__(self, dim:int, num_heads:int, 
                         img_size:int,
                         attn_drop:float=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        assert dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        ## q,k,v projection 
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.output_proj = nn.Linear(dim, dim)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(img_size)  # [1, 2*img_size-1, 2*img_size-1, 2]
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        relative_position_index = self._relative_position_index(img_size)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _relative_coords_table(self, img_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(img_size - 1), img_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(img_size - 1), img_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*img_size-1, 2*img_size-1, 2]
        relative_coords_table[:,:,:,0] /= img_size - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= img_size - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*img_size-1, 2*img_size-1, 2]

    def _relative_position_index(self, img_size: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(img_size)
        coords_w = torch.arange(img_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, img_size, img_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, img_size*img_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, img_size*img_size, img_size*img_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [img_size*img_size, img_size*img_size, 2]
        relative_coords[:, :, 0] += img_size - 1   # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += img_size - 1
        relative_coords[:, :, 0] *= 2 * img_size - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [img_size*img_size, img_size*img_size]
        return relative_position_index

    def forward(self, x):
        B,N,C = x.shape
        # Project the query, key, and value
        qkv = self.qkv_proj(x)  # [B, N, C*3]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        ## add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(     # type: ignore
                self.img_size * self.img_size, self.img_size * self.img_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.attn_dropout(F.softmax(attn, dim=-1)) 

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.output_proj(x)
        return x

class CrossAttention(nn.Module):
    """
    cross attention module.
    """
    def __init__(self, dim:int, num_heads:int, 
                            img_size:int, attn_drop:float=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        assert dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        ## q,k,v projection 
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)        
        self.output_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(img_size)  # [1, 2*img_size-1, 2*img_size-1, 2]
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        relative_position_index = self._relative_position_index(img_size)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _relative_coords_table(self, img_res: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(img_res - 1), img_res, dtype=torch.float32)
        relative_coords_w = torch.arange(-(img_res - 1), img_res, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*img_res-1, 2*img_res-1, 2]
        relative_coords_table[:,:,:,0] /= img_res - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= img_res - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*img_res-1, 2*img_res-1, 2]

    def _relative_position_index(self, img_res: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(img_res)
        coords_w = torch.arange(img_res)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, img_res, img_res]
        coords_flatten = torch.flatten(coords, 1)  # [2, img_res*img_res]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, img_res*img_res, img_res*img_res]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [img_res*img_res, img_res*img_res, 2]
        relative_coords[:, :, 0] += img_res - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += img_res - 1
        relative_coords[:, :, 0] *= 2 * img_res - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [img_res*img_res, img_res*img_res]
        return relative_position_index

    def forward(self, x, y):
        B, N, C = x.shape
        # Project the query, key, and value
        q_x = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        kv_y = self.kv_proj(y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N, head_dim]
        k_y, v_y = kv_y[0], kv_y[1]  # [B, num_heads, N, head_dim]
        attn = (q_x @ k_y.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        # attn = F.normalize(q_x, dim=-1) @ F.normalize(k_y, dim=-1).transpose(-2, -1) # [B, num_heads, N, N]
        # attn = attn * torch.clamp(self.logit_scale, max=np.log(100.)).exp()
        ## add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.img_size * self.img_size, self.img_size * self.img_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.attn_dropout(F.softmax(attn, dim=-1)) 
        x = (attn @ v_y).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.output_proj(x)
        return x   

class CrossAttention_dem(nn.Module):
    """
    cross attention module.
    """
    def __init__(self, dim:int, num_heads:int, 
                            img_size:int, attn_drop:float=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        assert dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        ## q,k,v projection 
        self.v_proj = nn.Linear(dim, dim)
        self.qk_proj = nn.Linear(dim, dim * 2)        
        self.output_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(img_size)     # [1, 2*img_size-1, 2*img_size-1, 2]
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        relative_position_index = self._relative_position_index(img_size)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _relative_coords_table(self, img_res: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(img_res - 1), img_res, dtype=torch.float32)
        relative_coords_w = torch.arange(-(img_res - 1), img_res, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*img_res-1, 2*img_res-1, 2]
        relative_coords_table[:,:,:,0] /= img_res - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= img_res - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*img_res-1, 2*img_res-1, 2]

    def _relative_position_index(self, img_res: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(img_res)
        coords_w = torch.arange(img_res)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, img_res, img_res]
        coords_flatten = torch.flatten(coords, 1)  # [2, img_res*img_res]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, img_res*img_res, img_res*img_res]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [img_res*img_res, img_res*img_res, 2]
        relative_coords[:, :, 0] += img_res - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += img_res - 1
        relative_coords[:, :, 0] *= 2 * img_res - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [img_res*img_res, img_res*img_res]
        return relative_position_index

    def forward(self, x, d):
        B, N, C = x.shape
        # Project the query, key, and value
        v_x = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        qk_d = self.qk_proj(d).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N, head_dim]
        q_d, k_d = qk_d[0], qk_d[1]  # [B, num_heads, N, head_dim]
        attn = (q_d @ k_d.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        # attn = F.normalize(q_x, dim=-1) @ F.normalize(k_d, dim=-1).transpose(-2, -1) # [B, num_heads, N, N]
        # attn = attn * torch.clamp(self.logit_scale, max=np.log(100.)).exp()
        ## add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.img_size * self.img_size, self.img_size * self.img_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0) 
        attn = self.attn_dropout(F.softmax(attn, dim=-1)) 
        x = (attn @ v_x).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.output_proj(x)
        return x   
    
class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA) or Shifted-Window MSA (SW-MSA)"""    
    def __init__(self, dim: int, 
                        window_size: int, 
                        num_heads: int, 
                        shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        assert dim % num_heads == 0,\
              "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(window_size)  # [window_size, window_size, 2]
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        relative_position_index = self._relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _relative_coords_table(self, window_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)    # [1, 2*window_size-1, 2*window_size-1, 2]
        relative_coords_table[:,:,:,0] /= window_size - 1     # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= window_size - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*window_size-1, 2*window_size-1, 2]

    def _relative_position_index(self, window_size: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size, window_size*window_size, 2]
        relative_coords[:, :, 0] += window_size - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]
        return relative_position_index

    def forward(self, windows_x, mask):
        B, N, C = windows_x.shape   
        # Generate Q, K, V
        qkv = self.qkv(windows_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask if provided (for shifted window)
        if mask is not None:
            nW = mask.shape[0] # Number of windows, [total_windows, window_size*window_size, window_size*window_size]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Cross_WindowAttention(nn.Module):
    """Window-based Multi-head Cross-Attention (W-MCA) or Shifted-Window MCA (SW-MCA)"""    
    def __init__(self, dim: int, 
                        window_size: int, 
                        num_heads: int, 
                        drop_attn: float = 0, 
                        proj_drop: float = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0,\
              "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj_drop = nn.Dropout(proj_drop)

        # Query, Key, Value projections
        # self.qkv = nn.Linear(dim, dim * 3)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(window_size)  # [window_size, window_size, 2]
        self.register_buffer("relative_coords_table", relative_coords_table)
        relative_position_index = self._relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

    def _relative_coords_table(self, window_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*window_size-1, 2*window_size-1, 2]
        relative_coords_table[:,:,:,0] /= window_size - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= window_size - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*window_size-1, 2*window_size-1, 2]

    def _relative_position_index(self, window_size: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size, window_size*window_size, 2]
        relative_coords[:, :, 0] += window_size - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]
        return relative_position_index

    def forward(self, windows_x: torch.Tensor, windows_y: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
        '''
        windows_x: [B, N, C], B: batch_size*num_windows, N: window_size*window_size, C: feature dimension,
        windows_y: [B, N, C], B: batch_size*num_windows, N: window_size*window_size, C: feature dimension, input feature map for key and value (from the other branch)
        mask: attention mask for shifted window        
        '''
        B, N, C = windows_x.shape        
        # Generate Q, K, V
        q_x = self.q(windows_x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [1, B, num_heads, N, head_dim]
        kv_y = self.kv(windows_y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N, head_dim]
        q_x, k_y, v_y = q_x[0], kv_y[0], kv_y[1]
        # Scaled dot-product attention
        attn = (q_x @ k_y.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(   # type: ignore
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        # Apply mask if provided (for shifted window)
        if mask is not None:
            nW = mask.shape[0] # Number of windows, [total_windows, window_size*window_size, window_size*window_size]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)        
        attn = F.softmax(attn, dim=-1)      
        attn = self.attn_drop(attn)  
        windows_x_att = (attn @ v_y).transpose(1, 2).reshape(B, N, C)  # Add skip connection with query
        windows_x_att = self.proj(windows_x_att)
        windows_x_att = self.proj_drop(windows_x_att)
        return windows_x_att


class Cross_WindowAttention_dem(nn.Module):
    """Window-based Multi-head Cross-Attention (W-MCA) or Shifted-Window MCA (SW-MCA)"""    
    def __init__(self, dim: int, 
                        window_size: int, 
                        num_heads: int, 
                        attn_drop: float = 0, 
                        proj_drop: float = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0,\
              "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Query, Key, Value projections
        # self.qkv = nn.Linear(dim, dim * 3)
        self.v = nn.Linear(dim, dim)
        self.qk = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(window_size)  # [window_size, window_size, 2]
        self.register_buffer("relative_coords_table", relative_coords_table)
        relative_position_index = self._relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

    def _relative_coords_table(self, window_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*window_size-1, 2*window_size-1, 2]
        relative_coords_table[:,:,:,0] /= window_size - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= window_size - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*window_size-1, 2*window_size-1, 2]

    def _relative_position_index(self, window_size: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size, window_size*window_size, 2]
        relative_coords[:, :, 0] += window_size - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]
        return relative_position_index

    def forward(self, windows_x: torch.Tensor, windows_y: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:  
        '''
        windows_x: [B, N, C], B: batch_size*num_windows, N: window_size*window_size, C: feature dimension,
        windows_d: [B, N, C], B: batch_size*num_windows, N: window_size*window_size, C: feature dimension, input feature map for key and value (from the other branch)
        mask: attention mask for shifted window        
        '''
        B, N, C = windows_x.shape        
        # Generate Q, K, V
        v_x = self.v(windows_x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  
        qk_y = self.qk(windows_y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  
        q_y, k_y, v_x = qk_y[0], qk_y[1], v_x[0]
        # Scaled dot-product attention
        attn = (q_y @ k_y.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(   # type: ignore
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        # Apply mask if provided (for shifted window)
        if mask is not None:
            nW = mask.shape[0] # Number of windows, [total_windows, window_size*window_size, window_size*window_size]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)        
        attn = F.softmax(attn, dim=-1)  
        attn = self.attn_drop(attn)  
        windows_x_att = (attn @ v_x).transpose(1, 2).reshape(B, N, C)  # Add skip connection with query
        windows_x_att = self.proj(windows_x_att)
        windows_x_att = self.proj_drop(windows_x_att)
        return windows_x_att

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA and SW-MSA"""
    def __init__(self, dim: int, 
                        num_heads: int, 
                        input_resolution: tuple[int, int],
                        window_size: int, 
                        shift_size: int = 0, 
                        mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Window attention
        self.window_attn = WindowAttention(dim, window_size, num_heads, shift_size) 

        # MLP
        self.mlp = MLP(dim, mlp_ratio)
        # Attention mask for SW-MSA
        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def _create_mask(self, input_resolution: tuple[int, int]) -> torch.Tensor:
        """Create mask for shifted window attention
        """
        H, W = input_resolution
        # Ensure dimensions are compatible
        shift_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:    
            for w in w_slices:    
                shift_mask[:, h, w, :] = cnt   # Assign a unique mask value to each region
                cnt += 1  
        shift_mask_windows = window_partition(shift_mask, self.window_size)  # [num_windows, window_size, window_size, 1]
        shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size) # [total_windows, window_size*window_size]
        attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # [total_windows, window_size*window_size, window_size*window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def _att(self, x: torch.Tensor) -> torch.Tensor:
        """Apply window attention to input feature map"""
        # B, H, W, C = x.shape
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C)  # Reshape to [B, H, W, C]
        # Cyclic shift for shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x        
        # Window partition
        x_windows = window_partition(shifted_x, self.window_size) # [total_windows*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # Window attention: W-MSA / SW-MSA
        attn_windows = self.window_attn(x_windows, self.attn_mask)        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x        
        x = x.view(B, H * W, C)  # Reshape back to [B, H*W, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: [B, H*W, C]'''
        shortcut = x  # [B, H*W, C]
        x = self.norm1(x)        
        # Attention
        x = self._att(x)  # [B, H*W, C]
        x = shortcut + x    # Residual connection after attention
        # FFN
        x = x + self.mlp(self.norm2(x))    # [B, H*W, C]     
        # x = x + self.norm2(self.mlp(x))    # [B, H*W, C]
        return x

class ViTransformerBlock(nn.Module):
    """Visual Transformer Block"""
    def __init__(self, dim: int,   
                        num_heads: int,  
                        img_size: int,  
                        mlp_ratio: int = 4, 
                        drop_attn: float = 0, 
                        drop_path: float = 0):  
        super().__init__()
        self.dim = dim  
        self.num_heads = num_heads  
        self.img_size = img_size
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        # cross attention
        self.cross_attn = Attention(dim, num_heads, img_size, drop_attn)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        # MLP
        self.mlp = MLP(dim, mlp_ratio)
        
    def forward(self, x: torch.Tensor):
        '''x: [B, H*W, C], y: [B, H*W, C]'''
        # Attention
        shortcut = x   # [B, H*W, C]
        x_norm = self.norm1(x) 
        # skip connection
        x_att = self.cross_attn(x_norm)
        x = shortcut + self.drop_path(x_att)    # Residual connection after attention
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))    # [B, H*W, C]     
        return x

class Cross_ViTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA and SW-MSA"""
    def __init__(self, dim: int,   
                        num_heads: int,  
                        img_size: int,  
                        mlp_ratio: int = 4, 
                        drop_attn: float = 0, 
                        drop_path: float = 0):  
        super().__init__()
        self.dim = dim  
        self.num_heads = num_heads  
        self.img_size = img_size
        # Layer normalization
        self.norm1_A = nn.LayerNorm(dim)
        self.norm1_B = nn.LayerNorm(dim)    
        # cross attention
        self.cross_attn_A = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_B = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_A = nn.LayerNorm(dim)
        self.norm2_B = nn.LayerNorm(dim)
        # MLP
        self.mlp_A = MLP(dim, mlp_ratio)
        self.mlp_B = MLP(dim, mlp_ratio)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''x: [B, H*W, C], y: [B, H*W, C]'''
        # Attention
        shortcut_A = x        # [B, H*W, C]
        shortcut_B = y        # [B, H*W, C]
        x_norm = self.norm1_A(x) 
        y_norm = self.norm1_B(y)  
        # skip connection
        x_att = self.cross_attn_A(x_norm, y_norm)   # [B, H*W, C], x: main(Q）, y: auxiliary(K, V)
        y_att = self.cross_attn_B(y_norm, x_norm)   # [B, H*W, C], y: main(Q), x: auxiliary(K, V)
        x = shortcut_A + self.drop_path_A(x_att)    # Residual connection after attention
        y = shortcut_B + self.drop_path_B(y_att)    # Residual connection after attention
        # FFN
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))    # [B, H*W, C]     
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))    # [B, H*W, C]     
        return x, y

class Cross_ViTransformerBlock_xyd(nn.Module):
    """ Swin Transformer Block with W-MSA and SW-MSA
    """
    def __init__(self, dim: int,   
                        num_heads: int,  
                        img_size: int,  
                        mlp_ratio: int = 4, 
                        drop_attn: float = 0, 
                        drop_path: float = 0):  
        super().__init__()
        self.dim = dim  
        self.num_heads = num_heads  
        self.img_size = img_size
        # Layer normalization
        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_y = nn.LayerNorm(dim)
        self.norm1_d = nn.LayerNorm(dim)

        self.w_x = nn.Parameter(torch.zeros(3))
        self.w_y = nn.Parameter(torch.zeros(3))
        self.w_d = nn.Parameter(torch.zeros(3))

        # cross attention        
        self.attn_x = Attention(dim, num_heads, img_size, drop_attn)
        self.attn_y = Attention(dim, num_heads, img_size, drop_attn)
        self.attn_d = Attention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_x_y = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_x_d = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_y_x = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_y_d = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_d_x = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_d_y = CrossAttention(dim, num_heads, img_size, drop_attn)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_x = nn.LayerNorm(dim)
        self.norm2_y = nn.LayerNorm(dim)
        self.norm2_d = nn.LayerNorm(dim)

        # MLP
        self.mlp_x = MLP(dim, mlp_ratio)
        self.mlp_y = MLP(dim, mlp_ratio)
        self.mlp_d = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor, d:torch.Tensor):
        '''x: [B, H*W, C], y: [B, H*W, C], z: [B, H*W, C]'''
        # Attention
        shortcut_x = x              # [B, H*W, C]
        shortcut_y = y              # [B, H*W, C]
        shortcut_d = d              # [B, H*W, C]
        x_norm = self.norm1_x(x)
        y_norm = self.norm1_y(y)
        d_norm = self.norm1_d(d)
        wx = torch.softmax(self.w_x, dim=0)
        wy = torch.softmax(self.w_y, dim=0)
        wd = torch.softmax(self.w_d, dim=0)

        # skip connection
        # x
        x_att0 = wx[0] * self.attn_x(x_norm)
        x_att1 = wx[1] * self.cross_attn_x_y(x_norm, y_norm)
        x_att2 = wx[2] * self.cross_attn_x_d(x_norm, d_norm)
        x = shortcut_x + self.drop_path(x_att0 + x_att1 + x_att2)

        # y
        y_att0 = wy[0] * self.attn_y(y_norm)
        y_att1 = wy[1] * self.cross_attn_y_x(y_norm, x_norm)
        y_att2 = wy[2] * self.cross_attn_y_d(y_norm, d_norm)
        y = shortcut_y + self.drop_path(y_att0 + y_att1 + y_att2)

        # d
        d_att0 = wd[0] * self.attn_d(d_norm) 
        d_att1 = wd[1] * self.cross_attn_d_x(d_norm, x_norm)
        d_att2 = wd[2] * self.cross_attn_d_y(d_norm, y_norm)
        d = shortcut_d + self.drop_path(d_att0 + d_att1 + d_att2)

        # FFN
        x = x + self.drop_path(self.mlp_x(self.norm2_x(x)))    # [B, H*W, C]     
        y = y + self.drop_path(self.mlp_y(self.norm2_y(y)))    # [B, H*W, C]     
        d = d + self.drop_path(self.mlp_d(self.norm2_d(d)))    # [B, H*W, C]    
        return x, y, d


class Cross_SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA and SW-MSA"""
    def __init__(self, dim: int,   
                        num_heads: int,  
                        input_resolution: tuple[int, int],  
                        window_size: int,   
                        shift_size: int = 0,  
                        mlp_ratio: int = 4,  
                        drop_attn: float = 0,
                        drop_path: float = 0.1):  
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer normalization
        self.norm1_A = nn.LayerNorm(dim)
        self.norm1_B = nn.LayerNorm(dim)
        
        # Window attention
        self.cross_window_attn_A = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_B = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = nn.LayerNorm(dim)
        self.norm2_B = nn.LayerNorm(dim)

        # MLP
        self.mlp_A = MLP(dim, mlp_ratio)
        self.mlp_B = MLP(dim, mlp_ratio)

        # Attention mask for SW-MSA
        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def _create_mask(self, input_resolution: tuple[int, int]) -> torch.Tensor:
        """Create mask for shifted window attention
        """
        H, W = input_resolution
        # Ensure dimensions are compatible
        shift_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:    
            for w in w_slices:    
                shift_mask[:, h, w, :] = cnt   # Assign a unique mask value to each region
                cnt += 1  
        shift_mask_windows = window_partition(shift_mask, self.window_size)  # [num_windows, window_size, window_size, 1]
        shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size) # [total_windows, window_size*window_size]
        attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # [total_windows, window_size*window_size, window_size*window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _att(self, x: torch.Tensor, y: torch.Tensor, cross_attn_module: nn.Module) -> torch.Tensor:
        """Apply window attention to input feature map"""
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)  # Reshape to [B, H, W, C]
        y = y.view(B, H, W, C)  # Reshape to [B, H, W, C]
        # Cyclic shift for shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x   
            shifted_y = y    
        # Window partition
        windows_x = window_partition(shifted_x, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_x = windows_x.view(-1, self.window_size * self.window_size, C)  # [total_windows*B, window_size*window_size, C]
        windows_y = window_partition(shifted_y, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_y = windows_y.view(-1, self.window_size * self.window_size, C)

        # Cross attention: W-MSA / SW-MSA 
        # (main information: x_windows, auxiliary information: y_windows)
        attn_windows_x = cross_attn_module(windows_x=windows_x, windows_y=windows_y,  mask=self.attn_mask)        
        # Merge windows
        attn_windows_x = attn_windows_x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_x, self.window_size, H, W)        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x        
        x = x.view(B, H * W, C)  # Reshape back to [B, H*W, C]
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        '''x: [B*n_window, window_h*window_w, C]'''
        # Attention
        shortcut_A = x     ## [B, H*W, C]
        shortcut_B = y     ## [B, H*W, C]
        x_norm = self.norm1_A(x)  
        y_norm = self.norm1_B(y)  
        # skip connection
        x_att = self._att(x_norm, y_norm, self.cross_window_attn_A)  # [B, H*W, C], x: main(Q）, y: auxiliary(K, V)
        y_att = self._att(y_norm, x_norm, self.cross_window_attn_B)  # [B, H*W, C], y: main(Q), x: auxiliary(K, V)
        # FFN
        x = shortcut_A + self.drop_path_A(x_att)    # Residual connection after attention
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))    # [B, H*W, C]     
        y = shortcut_B + self.drop_path_B(y_att)    # Residual connection after attention
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))    # [B, H*W, C]     
        return x, y

class Cross_SwinTransformerBlock_xyd(nn.Module):
    """Swin Transformer Block with W-MSA and SW-MSA"""
    def __init__(self,  dim: int,   
                        num_heads: int,  
                        input_resolution: tuple[int, int],  
                        window_size: int,   
                        shift_size: int = 0,  
                        mlp_ratio: int = 4,  
                        drop_attn: float = 0,
                        drop_path: float = 0.1):  
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Layer normalization
        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_y = nn.LayerNorm(dim)
        self.norm1_d = nn.LayerNorm(dim)
        self.w_x = nn.Parameter(torch.zeros(3))
        self.w_y = nn.Parameter(torch.zeros(3))
        self.w_d = nn.Parameter(torch.zeros(3))
        
        # Window attention
        self.window_attn_x_x = WindowAttention(dim, window_size, num_heads)
        self.window_attn_y_y = WindowAttention(dim, window_size, num_heads)
        self.window_attn_d_d = WindowAttention(dim, window_size, num_heads) 
        self.cross_window_attn_x_y = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_x_d = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_y_x = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_y_d = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_d_x = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)
        self.cross_window_attn_d_y = Cross_WindowAttention(dim, window_size, num_heads, drop_attn)

        self.norm2_x = nn.LayerNorm(dim)
        self.norm2_y = nn.LayerNorm(dim)
        self.norm2_d = nn.LayerNorm(dim)

        # MLP
        self.mlp_x = MLP(dim, mlp_ratio)
        self.mlp_y = MLP(dim, mlp_ratio)
        self.mlp_d = MLP(dim, mlp_ratio)

        # Attention mask for SW-MSA
        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def _create_mask(self, input_resolution: tuple[int, int]) -> torch.Tensor:
        """Create mask for shifted window attention
        """
        H, W = input_resolution
        # Ensure dimensions are compatible
        shift_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:    
            for w in w_slices:    
                shift_mask[:, h, w, :] = cnt   # Assign a unique mask value to each region
                cnt += 1  
        shift_mask_windows = window_partition(shift_mask, self.window_size)  # [num_windows, window_size, window_size, 1]
        shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size) # [total_windows, window_size*window_size]
        attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # [total_windows, window_size*window_size, window_size*window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _att_cross(self, x: torch.Tensor, y: torch.Tensor, cross_attn_module: nn.Module) -> torch.Tensor:
        """Apply window attention to input feature map"""
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)  # Reshape to [B, H, W, C]
        y = y.view(B, H, W, C)  # Reshape to [B, H, W, C]
        # Cyclic shift for shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x   
            shifted_y = y    
        # Window partition
        windows_x = window_partition(shifted_x, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_x = windows_x.view(-1, self.window_size * self.window_size, C)  # [total_windows*B, window_size*window_size, C]
        windows_y = window_partition(shifted_y, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_y = windows_y.view(-1, self.window_size * self.window_size, C)

        # Cross attention: W-MSA / SW-MSA 
        # (main information: x_windows, auxiliary information: y_windows)
        attn_windows_x = cross_attn_module(windows_x=windows_x, windows_y=windows_y, mask=self.attn_mask)        
        # Merge windows
        attn_windows_x = attn_windows_x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_x, self.window_size, H, W)        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x        
        x = x.view(B, H * W, C)  # Reshape back to [B, H*W, C]
        return x

    def _att_self(self, x: torch.Tensor, attn_module: nn.Module) -> torch.Tensor:
        """Apply window attention to input feature map"""
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)  # Reshape to [B, H, W, C]
        # Cyclic shift for shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x   
        # Window partition
        windows_x = window_partition(shifted_x, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_x = windows_x.view(-1, self.window_size * self.window_size, C)  # [total_windows*B, window_size*window_size, C]

        # Cross attention: W-MSA / SW-MSA 
        # (main information: x_windows, auxiliary information: y_windows)
        attn_windows_x = attn_module(windows_x=windows_x, mask=self.attn_mask)        
        # Merge windows
        attn_windows_x = attn_windows_x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_x, self.window_size, H, W)        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x        
        x = x.view(B, H * W, C)  # Reshape back to [B, H*W, C]
        return x
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor):
        '''x: [B*n_window, window_h*window_w, C]'''
        # Attention
        shortcut_x = x      ## [B, H*W, C]
        shortcut_y = y      ## [B, H*W, C]
        shortcut_d = d      ## [B, H*W, C]
        x_norm = self.norm1_x(x)  
        y_norm = self.norm1_y(y)  
        d_norm = self.norm1_d(d)  
        wx = torch.softmax(self.w_x, dim=0)
        wy = torch.softmax(self.w_y, dim=0)
        wd = torch.softmax(self.w_d, dim=0)

        # skip connection
        x_att0 = wx[0] * self._att_self(x_norm, self.window_attn_x_x)
        x_att1 = wx[1] * self._att_cross(x_norm, y_norm, self.cross_window_attn_x_y) 
        x_att2 = wx[2] * self._att_cross(x_norm, d_norm, self.cross_window_attn_x_d)  
        x = shortcut_x + self.drop_path(x_att0 + x_att1 + x_att2)

        y_att0 = wy[0] * self._att_self(y_norm, self.window_attn_y_y)
        y_att1 = wy[1] * self._att_cross(y_norm, x_norm, self.cross_window_attn_y_x) 
        y_att2 = wy[2] * self._att_cross(y_norm, d_norm, self.cross_window_attn_y_d)  
        y = shortcut_y + self.drop_path(y_att0 + y_att1 + y_att2)

        d_att0 = wd[0] * self._att_self(d_norm, self.window_attn_d_d)
        d_att1 = wd[1] * self._att_cross(d_norm, x_norm, self.cross_window_attn_d_x) 
        d_att2 = wd[2] * self._att_cross(d_norm, y_norm, self.cross_window_attn_d_y)  
        d = shortcut_d + self.drop_path(d_att0 + d_att1 + d_att2)        

        # # FFN
        x = x + self.drop_path(self.mlp_x(self.norm2_x(x)))    # [B, H*W, C]     
        y = y + self.drop_path(self.mlp_y(self.norm2_y(y)))    # [B, H*W, C]     
        d = d + self.drop_path(self.mlp_d(self.norm2_d(d)))    # [B, H*W, C]     
        return x, y, d

class SwinBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, 
                 dim, 
                 input_resolution: tuple[int, int], 
                 depth, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size
        assert 0 <= self.window_size <= min(self.input_resolution), \
            f"Window size {self.window_size} is larger than input resolution {self.input_resolution}"
        assert all(i % window_size == 0 for i in self.input_resolution), \
            f"Input resolution {self.input_resolution} must be divisible by window size {window_size}"
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, 
                                 num_heads=num_heads,
                                 input_resolution=self.input_resolution,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 )
                for i in range(depth)])

    def forward(self, x):
        'x, y: [B, C, H, W], output: [B, C, H, W]'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()  # [B, C, H, W] -> [B, H*W, C]
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return x


class VitBasicLayer(nn.Module):
    """ A basic ViTransformer layer for one stage.
    """
    def __init__(self,
                 dim,
                 img_size,
                 depth,
                 num_heads,
                 mlp_ratio=4,
                 drop_attn=0.,
                 drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.depth = depth
        # linearly increasing stochastic-depth rate along the depth dimension
        # dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        # separate blocks for each modality pair to allow specialization
        self.blocks = nn.ModuleList([
                ViTransformerBlock(dim=dim,
                                     num_heads=num_heads,
                                     img_size=img_size,
                                     mlp_ratio=mlp_ratio,
                                     drop_attn=drop_attn,
                                     drop_path=drop_path)
                            for i in range(depth)])

        # final per-branch normalization before fusion
        self.norm_x = nn.LayerNorm(dim)

    def forward(self, x):
        'x: [B, C, H, W], output: [B, C, H, W]'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()  ## [B, C, H, W] -> [B, H*W, C]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_x(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return x
    
class Cross_Vit_Fusion(nn.Module):
    """ A basic ViTransformer layer for one stage.
    """
    def __init__(self,
                 dim,
                 img_size,
                 depth,
                 num_heads,
                 mlp_ratio=4,
                 drop_attn=0.,
                 drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.depth = depth
        # linearly increasing stochastic-depth rate along the depth dimension
        # dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        # separate blocks for each modality pair to allow specialization
        self.blocks = nn.ModuleList([
            Cross_ViTransformerBlock_xyd(dim=dim,
                                     num_heads=num_heads,
                                     img_size=img_size,
                                     mlp_ratio=mlp_ratio,
                                     drop_attn=drop_attn,
                                     drop_path=drop_path)
                        for i in range(depth)])

    def forward(self, x, y, z):
        'x, y, z: [B, C, H, W], output: [B, 3*C, H, W]'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()  ## [B, C, H, W] -> [B, H*W, C]
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        z = z.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y, z = blk(x,y,z)        
        x_fus = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]   
        y_fus = y.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]    
        z_fus = z.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]   
        return x_fus, y_fus, z_fus

class Cross_SwinBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self, 
                 dim, 
                 input_resolution: tuple[int, int], 
                 depth, 
                 num_heads, 
                 window_size,
                 drop_attn=0.,
                 drop_path=0.1,
                 mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size
        assert 0 <= self.window_size <= min(self.input_resolution), \
            f"Window size {self.window_size} is larger than input resolution {self.input_resolution}"
        assert all(i % window_size == 0 for i in self.input_resolution), \
            f"Input resolution {self.input_resolution} must be divisible by window size {window_size}"
        # build blocks
        self.blocks = nn.ModuleList([
            Cross_SwinTransformerBlock_xyd(dim=dim, 
                                 num_heads=num_heads,
                                 input_resolution=self.input_resolution,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 drop_attn=drop_attn,
                                 drop_path=drop_path
                                 )
                    for i in range(depth)])  

    def forward(self, x, y, d):
        'x, y, z: [B, C, H, W], output: [B, C, H, W]'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()  # [B, C, H, W] -> [B, H*W, C]
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        d = d.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y, d = blk(x, y, d)
        x_fus = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        y_fus = y.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        d_fus = d.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return x_fus, y_fus, d_fus

class PatchExpand(nn.Module):
    """ Patch Expand Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.conv = conv(dim, 4*dim, kernel_size=1, stride=1, padding=0)        
        self.norm = norm_layer(dim)

    def forward(self, x):
        """ x: [B, C, H, W] -> [B, C, 2H, 2W] """ 
        B, C, H, W = x.shape
        x = self.conv(x)  # [B, 4C, H, W]
        x = x.view(B, C, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, 2*H, 2*W)
        x = self.norm(x)    # [B, C, 2H, 2W]
        return x

class u3net_cross_fusion_(nn.Module):
    def __init__(self, backbone_name='resnet34',
                        pretrained=True):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super().__init__()
        # self.k = nn.Parameter(torch.tensor(55.0/(8848+420)))      # m/degree
        # self.ref_phi = 45.0 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)        
        ## encoder part 
        self.encoder_opt = timm.create_model(backbone_name,  
                                        features_only=True,  
                                        in_chans=3,  
                                        pretrained=pretrained)
        self.encoder_nir = timm.create_model(backbone_name,   
                                        features_only=True,   
                                        in_chans=3,   
                                        pretrained=pretrained)
        self.encoder_dem = timm.create_model(backbone_name,
                                        features_only=True,
                                        in_chans=1,
                                        pretrained=pretrained)

        self.out_channels = self.encoder_opt.feature_info.channels()   ## from shallow to deep  # type: ignore
        print("Output channels from encoder stages:", self.out_channels)
        # self.decode_channels = [64, 64, 64, 64, 32]   ##  from deep to shallow
        self.decode_channels = [96, 96, 96, 96, 48]   ##  from deep to shallow
        # self.decode_channels = [128, 128, 128, 128, 64]  # decoder channels for each stage
        # self.decode_channels = [128, 128, 128, 128, 64]  # decoder channels for each stage, same as encoder channels
        self.cross_layers_4 = Cross_Vit_Fusion(dim=self.out_channels[4],
                                            img_size=16,
                                            depth=4,
                                            num_heads=4,
                                            mlp_ratio=2,
                                            drop_attn=0.,
                                            drop_path=0.1)
        # self.cross_layers_3 = Cross_Vit_Fusion(dim=self.out_channels[3],
        #                                     img_size=32,
        #                                     depth=4,
        #                                     num_heads=4,
        #                                     mlp_ratio=2,
        #                                     drop_attn=0.1,
        #                                     drop_path=0.2)
        self.cross_layers_3 = Cross_SwinBasicLayer(dim=self.out_channels[3],
                                            input_resolution=(32, 32),
                                            depth=2,
                                            num_heads=4,
                                            window_size=8,
                                            mlp_ratio=2,
                                            drop_attn=0.,
                                            drop_path=0.1)
        self.cross_layers_2 = Cross_SwinBasicLayer(dim=self.out_channels[2],   # layer 2: 64x64, 40ch
                                            input_resolution=(64, 64),
                                            depth=2,
                                            num_heads=4,
                                            window_size=8, 
                                            mlp_ratio=2,
                                            drop_attn=0.,
                                            drop_path=0.1,
                                            )

        ## decoder part (fused features), from deep to shallow
        self.DecoderBlocks = nn.ModuleList([
                conv(self.out_channels[4]*3, self.decode_channels[0]),   # layer 4
                conv(self.out_channels[3]*3+self.decode_channels[0], self.decode_channels[1]), # layer 3
                conv(self.out_channels[2]*3+self.decode_channels[1], self.decode_channels[2]), # layer 2
                conv(self.out_channels[1]*3+self.decode_channels[2], self.decode_channels[3]), # layer 1
                conv(self.out_channels[0]*3+self.decode_channels[3], self.decode_channels[4])  # layer 0
                ])
        self.outp = nn.Sequential(
                        nn.Conv2d(self.decode_channels[4], 1, kernel_size=3, padding=1),
                        )
        ## auxiliary outputs for deep supervision
        # self.aux1 = nn.Conv2d(self.decode_channels[3], 1, 3, padding=1)  # auxiliary output at layer 1
        # self.aux2 = nn.Conv2d(self.decode_channels[2], 1, 3, padding=1)  # auxiliary output at layer 2
        self.aux2 = nn.Conv2d(self.out_channels[2]*3, 1, 3, padding=1)  # auxiliary output at layer 3
        self.aux3 = nn.Conv2d(self.out_channels[3]*3, 1, 3, padding=1)  # auxiliary output at layer 3
        self.aux4 = nn.Conv2d(self.out_channels[4]*3, 1, 3, padding=1)  # auxiliary output at layer 4

    def _dem_geometry(self, x_dem):
        gy, gx = torch.gradient(x_dem, dim=(2, 3))
        slope = torch.sqrt(gx**2 + gy**2 + 1e-8)
        aspect = torch.atan2(gy, gx + 1e-8)
        slope = slope / (slope.amax(dim=(2,3), keepdim=True) + 1e-8)
        aspect_sin = torch.sin(aspect)     # 周期连续编码
        aspect_cos = torch.cos(aspect)
        return torch.cat([x_dem, slope, aspect_sin, aspect_cos], dim=1)  # [B,4,H,W]

    def forward(self, x, lat=None):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_opt = x[:, :3, :, :]
        x_nir = x[:, 3:6, :, :]
        x_dem = x[:, 6:, :, :]
        ## dem adjustment
        if lat is not None:
            pass     ## not implemented yet
            # adj = self.k * (torch.abs(lat) - self.ref_phi)   # lat: [B]
            # x_dem = x_dem + adj.view(-1, 1, 1, 1)
 
        ## encoder part
        feas_opt = self.encoder_opt(x_opt)  # list of features from encoder branch 1
        feas_nir = self.encoder_nir(x_nir)  # list of features from encoder branch 2
        feas_dem = self.encoder_dem(x_dem)  # list of features from encoder branch 3

        # layer 4
        fea_opt, fea_nir, fea_dem = feas_opt[-1], feas_nir[-1], feas_dem[-1]   #  [B, C, 16, 16] 
        aux4_out = self.aux4(torch.cat([fea_opt, fea_nir, fea_dem], dim=1))  # auxiliary output at layer 4
        fea_opt, fea_nir, fea_dem = self.cross_layers_4(fea_opt, fea_nir, fea_dem)   # cross attention fusion of features from three branches
        fea_fus_4 = torch.cat([fea_opt, fea_nir, fea_dem], dim=1)  # concat features from three branches
        fea_fus_4 = self.DecoderBlocks[0](fea_fus_4)      # fused features through decoder    
        fea_fus_4 = self.up2(fea_fus_4)  # upsample to match next skip connection

        # layer 3
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-2], feas_nir[-2], feas_dem[-2]        
        aux3_out = self.aux3(torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1))  # auxiliary output at layer 4
        skip_fea_opt, skip_fea_nir, skip_fea_dem = self.cross_layers_3(skip_fea_opt, skip_fea_nir, skip_fea_dem)   # cross attention fusion of skip features
        fea_fus_3 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        fea_fus_3 = torch.cat([fea_fus_4, fea_fus_3], dim=1)  # concat skip features
        fea_fus_3 = self.DecoderBlocks[1](fea_fus_3)  ## decode fused features
        fea_fus_3 = self.up2(fea_fus_3)               ## upsample for next stage

        # layer 2:
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-3], feas_nir[-3], feas_dem[-3]
        aux2_out = self.aux2(torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1))  # auxiliary output at layer 4
        skip_fea_opt, skip_fea_nir, skip_fea_dem = self.cross_layers_2(skip_fea_opt, skip_fea_nir, skip_fea_dem)     
        fea_fus_2 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        fea_fus_2 = torch.cat([fea_fus_3, fea_fus_2], dim=1)  # concat skip features
        fea_fus_2 = self.DecoderBlocks[2](fea_fus_2)  # decode fused features
        fea_fus_2 = self.up2(fea_fus_2)  # upsample for next stage

        # layer 1:
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-4], feas_nir[-4], feas_dem[-4]
        # fea_fus_1 = torch.cat([fea_fus_2, fea_fus_1], dim=1)  # concat skip features
        # skip_fus_1 = self.cross_layers_1(skip_fea_opt, skip_fea_nir, skip_fea_dem)  # windowed cross-modal fusion -> [B, 3C, 128, 128]
        # skip_fea_opt, skip_fea_nir, skip_fea_dem = self.ssf_fusion_1(skip_fea_opt, skip_fea_nir, skip_fea_dem)  # cross fusion of features at layer 1
        skip_fus_1 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        fea_fus_1 = torch.cat([fea_fus_2, skip_fus_1], dim=1)  # concat skip features
        fea_fus_1 = self.DecoderBlocks[3](fea_fus_1)  # decode fused features
        fea_fus_1 = self.up2(fea_fus_1)    ## upsample for next stage

        # layer 0:   
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-5], feas_nir[-5], feas_dem[-5]
        # fea_fus_0 = self.cross_layers_0(skip_fea_opt, skip_fea_nir, skip_fea_dem)  # cross attention fusion of skip features
        # fea_fus_0 = torch.cat([fea_fus_1, fea_fus_0], dim=1)  # concat skip features
        # skip_fea_opt, skip_fea_nir, skip_fea_dem = self.ssf_fusion_0(skip_fea_opt, skip_fea_nir, skip_fea_dem)  # cross fusion of features at layer 0
        fea_fus_0 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)  # concat skip features
        fea_fus_0 = torch.cat([fea_fus_1, fea_fus_0], dim=1)  # concat skip features
        fea_fus_0 = self.DecoderBlocks[4](fea_fus_0)  # decode fused features
        # fea_fus_0 = self.decode0(fea_fus_0)   
        fea_fus_0 = self.up2(fea_fus_0)  # upsample for next stage   
        logit = self.outp(fea_fus_0)    # 1x512x512    
        return logit, aux4_out, aux3_out, aux2_out # main output, auxiliary outputs for deep supervision
        # return logit

if __name__ == '__main__':
    model = u3net_cross_fusion_(
                            # backbone_name='resnet34', 
                            backbone_name='efficientnet_b0',
                            pretrained=True)
    x = torch.randn(2, 7, 512, 512)  # batch_size=2, num_bands=7, H=W=512
    out = model(x)
    print(out.shape)  # should be (2, 1, 512, 512)

