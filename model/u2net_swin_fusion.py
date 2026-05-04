'''
author: xin luo
create: 2026.4.24
des: UNet-like swin transformer model for segmentation task
ref:(1) https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    (2) https://github.com/amarbit/SwinTransformerFromScratch
# undo: use einops to simplify tensor reshaping and permutation 
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from timm.layers.helpers import to_2tuple


def conv(in_channels, out_channels, 
         kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        # nn.BatchNorm2d(out_channels),
        # nn.GroupNorm(1, out_channels),  # group normalization  
        # GlobalBatchNorm2d(out_channels),  # global batch normalization      
        nn.ReLU(inplace=True)
        )

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""    
    def __init__(self, img_size: int = 224, 
                            patch_size: int = 4, 
                            in_channels: int = 3, 
                            embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Project patches to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape  
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model's expected size ({self.img_size}*{self.img_size})."
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
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
        self.register_buffer("relative_coords_table", relative_coords_table)
        relative_position_index = self._relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

    def _relative_coords_table(self, window_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size - 1), self.window_size, dtype=torch.float32)
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

    def forward(self, x, mask):
        B, N, C = x.shape        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
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
    
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0,\
              "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
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
        x: [B, N, C], input feature map for query
        y: [B, N, C], input feature map for key and value (from the other branch)
        mask: attention mask for shifted window        
        '''
        B, N, C = windows_x.shape        
        # Generate Q, K, V
        q = self.q(windows_x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [1, B, num_heads, N, head_dim]
        kv = self.kv(windows_y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N, head_dim]
        q, k, v = q[0], kv[0], kv[1]
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
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
        windows_x_att = (attn @ v).transpose(1, 2).reshape(B, N, C)
        windows_x_att = self.proj(windows_x_att)
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
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

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
        # x = x + self.mlp(self.norm2(x))    # [B, H*W, C]     
        x = x + self.norm2(self.mlp(x))    # [B, H*W, C]
        return x


class Cross_SwinTransformerBlock(nn.Module):
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
        self.norm1_A = nn.LayerNorm(dim)
        self.norm1_B = nn.LayerNorm(dim)
        
        # Window attention
        self.cross_window_attn = Cross_WindowAttention(dim, window_size, num_heads)
        self.norm2_A = nn.LayerNorm(dim)
        self.norm2_B = nn.LayerNorm(dim)
        # MLP
        self.mlp_A = MLP(dim, mlp_ratio)
        self.mlp_B = MLP(dim, mlp_ratio)
        # Attention mask for SW-MSA
        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

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
    
    
    def _att(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        windows_x = windows_x.view(-1, self.window_size * self.window_size, C)
        windows_y = window_partition(shifted_y, self.window_size) # [total_windows*B, window_size, window_size, C]
        windows_y = windows_y.view(-1, self.window_size * self.window_size, C)

        # Cross attention: W-MSA / SW-MSA 
        # (main information: x_windows, auxiliary information: y_windows)
        attn_windows_x = self.cross_window_attn(windows_x=windows_x, 
                                            windows_y=windows_y, mask=self.attn_mask)        
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
        '''x: [B, w_h*w_w, C]'''
        shortcut_A = x   # [B, H*W, C]
        shortcut_B = y   # [B, H*W, C]
        x = self.norm1_A(x)        
        y = self.norm1_B(y)
        # Attention
        x = self._att(x, y)  # [B, H*W, C]
        y = self._att(y, x)  # [B, H*W, C]
        x = shortcut_A + x    # Residual connection after attention
        y = shortcut_B + y    # Residual connection after attention
        # FFN
        x = x + self.norm2_A(self.mlp_A(x))    # [B, H*W, C]     
        y = y + self.norm2_B(self.mlp_B(y))    # [B, H*W, C]     
        return x, y


class BasicLayer(nn.Module):
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

class Cross_BasicLayer(nn.Module):
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
            Cross_SwinTransformerBlock(dim=dim, 
                                 num_heads=num_heads,
                                 input_resolution=self.input_resolution,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 )
                    for i in range(depth)])
        self.conv1x1 = conv(dim*2, dim, kernel_size=1, stride=1, padding=0)  # convolution layer for feature fusion after cross attention
    def forward(self, x, y):
        'x, y: [B, C, H, W], output: [B, C, H, W]'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()  # [B, C, H, W] -> [B, H*W, C]
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y = blk(x, y)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        y = y.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        # fus_xy = x + y
        fus_xy = torch.cat([x, y], dim=1)  # [B, 2*C, H, W]
        fus_xy = self.conv1x1(fus_xy)  # [B, C, H, W]
        return fus_xy

class u2net_swin_fusion(nn.Module):
    def __init__(self, num_bands_b1, num_bands_b2, image_size=512, window_size=8):
        '''
        num_bands_b1: number of bands for branch 1 (e.g., scene image)
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super(u2net_swin_fusion, self).__init__()
        self.num_bands_b1 = num_bands_b1
        self.num_bands_b2 = num_bands_b2
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer
        ## encoder part
        ### branch 1
        self.encoder_conv1_b1 = conv(self.num_bands_b1, 48)        
        self.encoder_conv2_b1 = conv(48, 96)
        self.encoder_conv3_b1 = conv(96, 96)
        self.encoder_conv4_b1 = conv(96, 96)
        ### branch 2
        self.encoder_conv1_b2 = conv(self.num_bands_b2, 48)  # for DEM, no batch norm
        self.encoder_conv2_b2 = conv(48, 96)
        self.encoder_conv3_b2 = conv(96, 96)
        self.encoder_conv4_b2 = conv(96, 96)
        ## fusion 
        self.swin_fusion1 = Cross_BasicLayer(dim=48, input_resolution=(image_size//2, image_size//2), 
                                             depth=2, num_heads=4, window_size=window_size) # 256  
        self.swin_fusion2 = Cross_BasicLayer(dim=96, input_resolution=(image_size//4, image_size//4), 
                                             depth=2, num_heads=4, window_size=window_size) # 128
        self.swin_fusion3 = Cross_BasicLayer(dim=96, input_resolution=(image_size//8, image_size//8), 
                                             depth=2, num_heads=4, window_size=window_size) # 64
        self.swin_fusion4 = Cross_BasicLayer(dim=96, input_resolution=(image_size//16, image_size//16), 
                                             depth=2, num_heads=4, window_size=window_size) # 32

        ## decoder part (fused features)
        self.decoder4 = conv(96, 96)    
        self.decoder3 = conv(96+96, 96)    
        self.decoder2 = conv(96+96, 96)    
        self.decoder1 = conv(48+96, 96)    
        # self.decoder4 = BasicLayer(dim=96, input_resolution=(image_size//16, image_size//16), 
        #                                         depth=2, num_heads=4, window_size=window_size)
        # self.decoder3 = BasicLayer(dim=192, input_resolution=(image_size//8, image_size//8), 
        #                                         depth=2, num_heads=4, window_size=window_size)
        # self.decoder2 = BasicLayer(dim=192, input_resolution=(image_size//4, image_size//4), 
        #                                         depth=2, num_heads=4, window_size=window_size)
        self.outp = nn.Sequential(
                        nn.Conv2d(96, 1, kernel_size=3, padding=1)) 
    
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Linear, nn.Conv2d)): 
    #         trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)            
    #         if m.bias is not None:     
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):            
    #         if m.weight is not None: 
    #             nn.init.ones_(m.weight)            
    #         if m.bias is not None:                
    #             nn.init.zeros_(m.bias)

    def forward(self, x):       ## input size: 7x256x256
        '''
        x: input tensor
        '''
        x_b1, x_b2 = x[:, :self.num_bands_b1, :, :], x[:, self.num_bands_b1:, :, :]
        ## encoder part
        ### branch 1 (scene image)
        x1_b1 = self.encoder_conv1_b1(x_b1)   #           
        x1_b1 = F.avg_pool2d(input=x1_b1, kernel_size=2) #  size: 1/2
        x2_b1 = self.encoder_conv2_b1(x1_b1)               
        x2_b1 = F.avg_pool2d(input=x2_b1, kernel_size=2) #  size: 1/4
        x3_b1 = self.encoder_conv3_b1(x2_b1)               
        x3_b1 = F.avg_pool2d(input=x3_b1, kernel_size=2) #  size: 1/8
        x4_b1 = self.encoder_conv4_b1(x3_b1)              
        x4_b1 = F.avg_pool2d(input=x4_b1, kernel_size=2) #  size: 1/16
        ### branch 2 (DEM)
        x1_b2 = self.encoder_conv1_b2(x_b2)              
        x1_b2 = F.avg_pool2d(input=x1_b2, kernel_size=2)  #  size: 1/2
        x2_b2 = self.encoder_conv2_b2(x1_b2)              
        x2_b2 = F.avg_pool2d(input=x2_b2, kernel_size=2) #  size: 1/4 
        x3_b2 = self.encoder_conv3_b2(x2_b2)              
        x3_b2 = F.avg_pool2d(input=x3_b2, kernel_size=2) #  size: 1/8
        x4_b2 = self.encoder_conv4_b2(x3_b2)              
        x4_b2 = F.avg_pool2d(input=x4_b2, kernel_size=2) #  size: 1/16, 512->32

        ## fusion part (Swin Transformer blocks)
        x1_fus = self.swin_fusion1(x1_b1, x1_b2)  # 48*2
        x2_fus = self.swin_fusion2(x2_b1, x2_b2)  # 96*2
        x3_fus = self.swin_fusion3(x3_b1, x3_b2)  # 96*2
        x4_fus = self.swin_fusion4(x4_b1, x4_b2)  # 96*2

        ## decoder part
        x4_decoder = self.decoder4(x4_fus)   ##
        x3_decoder = self.decoder3(torch.cat([self.up(x4_decoder), x3_fus], dim=1))  
        x2_decoder = self.decoder2(torch.cat([self.up(x3_decoder), x2_fus], dim=1))  
        x1_decoder = self.decoder1(torch.cat([self.up(x2_decoder), x1_fus], dim=1))  

        x1_decoder = self.up(x1_decoder)  # up to original size
        logit = self.outp(x1_decoder)  
        return logit

if __name__ == "__main__":
    model = u2net_swin_fusion(num_bands_b1=6, num_bands_b2=1, 
                                    image_size=448, window_size=7)
    x = torch.randn(2, 7, 448, 448)  # batch size of 2
    logit = model(x)
    print(logit.shape) 

