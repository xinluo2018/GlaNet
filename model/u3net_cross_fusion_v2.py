'''
author: xin luo
create: 2026.7.4
des:
(1) v2 of u3net_cross_fusion_: triple-branch (opt/nir/dem) U-Net with
    cross-ViT bottleneck fusion, cleaned up from u3net_cross_fusion_.py:
    - removed the never-used self-attention modules (attn_x/y/d) inside
      Cross_ViTransformerBlock_xyd (dead parameters in v1);
    - kept state_dict key names compatible with u3net_cross_fusion_ so the
      0.952 checkpoint can be partially loaded (extra keys are discarded).
(2) optional windowed cross-modal fusion at shallow skips (layer 3/2),
    off by default (shallow_fusion=False).
(3) 2026.7.4 r3: bottleneck fusion switched from six pairwise CrossAttention
    modules (softmax-weighted sum, Cross_ViTransformerBlock_xyd) to ONE joint
    cross-attention per branch (Joint_CrossViTBlock_xyd): the query branch
    attends to the concatenated tokens of the other two modalities within a
    single softmax, so the context modalities compete for attention mass
    token-by-token instead of being mixed with fixed learned scalars.
    ~1/3 fewer fusion params (whole model shrinks), counteracting the
    confirmed overfitting (train mIoU ~0.975 vs val ~0.950).
    RESULT (r3 run, 600ep): best raw 0.9514 / ema 0.9512 — slightly BELOW the
    pairwise baseline (r2: 0.9521), so cross_layers_4 was reverted to
    Cross_Vit_Fusion; Joint_* classes kept below for the record.
    references:
      - TokenFusion: Wang et al., "Multimodal Token Fusion for Vision
        Transformers", CVPR 2022.
      - CMNeXt: Zhang et al., "Delivering Arbitrary-Modal Semantic
        Segmentation", CVPR 2023 (hub2fuse).
      - GeminiFusion: Jia et al., "Efficient Pixel-wise Multimodal Fusion for
        Vision Transformer", ICML 2024 (exhaustive pairwise cross-attention
        among modalities is largely redundant).
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
        nn.ReLU(inplace=True)
        )

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
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  ## [num_windows*B, window_size, window_size, C]

def window_reverse(windows: torch.Tensor,
                    window_size: int, H: int, W: int) -> torch.Tensor:
    """ Reverse window partition """
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def _relative_coords_table(size: int) -> torch.Tensor:
    """Relative coordinate table (log-scaled) for continuous position bias"""
    relative_coords_h = torch.arange(-(size - 1), size, dtype=torch.float32)
    relative_coords_w = torch.arange(-(size - 1), size, dtype=torch.float32)
    table = torch.stack(
        torch.meshgrid([relative_coords_h,
                        relative_coords_w], indexing='ij')
                        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*size-1, 2*size-1, 2]
    table[:, :, :, 0] /= size - 1  # Normalize to [-1, 1]
    table[:, :, :, 1] /= size - 1
    table *= 8  # Scale to increase resolution of relative position bias
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    return table

def _relative_position_index(size: int) -> torch.Tensor:
    """Relative position index for each token pair inside a size x size grid"""
    coords_h = torch.arange(size)
    coords_w = torch.arange(size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, size, size]
    coords_flatten = torch.flatten(coords, 1)  # [2, size*size]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += size - 1
    relative_coords[:, :, 1] += size - 1
    relative_coords[:, :, 0] *= 2 * size - 1
    relative_position_index = relative_coords.sum(-1)  # [size*size, size*size]
    return relative_position_index


class CrossAttention(nn.Module):
    """ cross attention module (q from x, k/v from y), full attention. """
    def __init__(self, dim: int, num_heads: int,
                            img_size: int, attn_drop: float = 0):
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
        self.register_buffer("relative_coords_table", _relative_coords_table(img_size), persistent=False)
        self.register_buffer("relative_position_index", _relative_position_index(img_size), persistent=False)

    def forward(self, x, y):
        B, N, C = x.shape
        q_x = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_y = self.kv_proj(y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_y, v_y = kv_y[0], kv_y[1]
        attn = (q_x @ k_y.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        ## add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.img_size * self.img_size, self.img_size * self.img_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        x = (attn @ v_y).transpose(1, 2).reshape(B, N, C)
        x = self.output_proj(x)
        return x


class Cross_ViTransformerBlock_xyd(nn.Module):
    """ Three-branch cross-attention transformer block (x/y/d).
    v2: dead self-attention modules removed; each branch fuses the other
    two branches via learnable softmax weights (w_*, 2 terms per branch).
    r12: per-branch SELF-attention added back as a third residual term
    (intra-modal + inter-modal attention, cf. MulT, Tsai et al., ACL 2019;
    SwinFusion intra-/inter-domain units, Ma et al., IEEE/CAA JAS 2022).
    Their output projections are ZERO-INITIALIZED: a warm-started model is
    numerically identical at step 0 and the branch only grows if useful.
    """
    def __init__(self, dim: int,
                        num_heads: int,
                        img_size: int,
                        mlp_ratio: int = 4,
                        drop_attn: float = 0,
                        drop_path: float = 0,
                        mod_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        # r5: modality dropout — per-sample, a modality is dropped as cross-
        # attention source (ModDrop, Neverova et al., TPAMI 2016; ShaSpec,
        # Wang et al., CVPR 2023); inverted scaling, inference unchanged.
        self.mod_drop = mod_drop
        # Layer normalization
        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_y = nn.LayerNorm(dim)
        self.norm1_d = nn.LayerNorm(dim)

        self.w_x = nn.Parameter(torch.zeros(2))
        self.w_y = nn.Parameter(torch.zeros(2))
        self.w_d = nn.Parameter(torch.zeros(2))

        # cross attention
        self.cross_attn_x_y = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_x_d = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_y_x = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_y_d = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_d_x = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.cross_attn_d_y = CrossAttention(dim, num_heads, img_size, drop_attn)

        # r12: intra-modal self-attention (q and k/v from the same branch)
        self.self_attn_x = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.self_attn_y = CrossAttention(dim, num_heads, img_size, drop_attn)
        self.self_attn_d = CrossAttention(dim, num_heads, img_size, drop_attn)
        for m in (self.self_attn_x, self.self_attn_y, self.self_attn_d):
            nn.init.zeros_(m.output_proj.weight)
            nn.init.zeros_(m.output_proj.bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_x = nn.LayerNorm(dim)
        self.norm2_y = nn.LayerNorm(dim)
        self.norm2_d = nn.LayerNorm(dim)

        # MLP
        self.mlp_x = MLP(dim, mlp_ratio)
        self.mlp_y = MLP(dim, mlp_ratio)
        self.mlp_d = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor):
        '''x, y, d: [B, H*W, C]'''
        shortcut_x = x
        shortcut_y = y
        shortcut_d = d
        x_norm = self.norm1_x(x)
        y_norm = self.norm1_y(y)
        d_norm = self.norm1_d(d)
        wx = torch.softmax(self.w_x, dim=0)
        wy = torch.softmax(self.w_y, dim=0)
        wd = torch.softmax(self.w_d, dim=0)

        # modality dropout masks: m_* gates modality * as attention SOURCE
        if self.training and self.mod_drop > 0.:
            keep = 1. - self.mod_drop
            m = torch.bernoulli(torch.full((3, x.shape[0], 1, 1), keep,
                                           device=x.device, dtype=x.dtype)) / keep
            m_x, m_y, m_d = m[0], m[1], m[2]
        else:
            m_x = m_y = m_d = 1.

        x_att1 = wx[0] * self.cross_attn_x_y(x_norm, y_norm) * m_y
        x_att2 = wx[1] * self.cross_attn_x_d(x_norm, d_norm) * m_d
        x = shortcut_x + self.drop_path(x_att1 + x_att2 + self.self_attn_x(x_norm, x_norm))

        y_att1 = wy[0] * self.cross_attn_y_x(y_norm, x_norm) * m_x
        y_att2 = wy[1] * self.cross_attn_y_d(y_norm, d_norm) * m_d
        y = shortcut_y + self.drop_path(y_att1 + y_att2 + self.self_attn_y(y_norm, y_norm))

        d_att1 = wd[0] * self.cross_attn_d_x(d_norm, x_norm) * m_x
        d_att2 = wd[1] * self.cross_attn_d_y(d_norm, y_norm) * m_y
        d = shortcut_d + self.drop_path(d_att1 + d_att2 + self.self_attn_d(d_norm, d_norm))

        # FFN
        x = x + self.drop_path(self.mlp_x(self.norm2_x(x)))
        y = y + self.drop_path(self.mlp_y(self.norm2_y(y)))
        d = d + self.drop_path(self.mlp_d(self.norm2_d(d)))
        return x, y, d


class Cross_Vit_Fusion(nn.Module):
    """ A stack of three-branch cross-ViT blocks for one stage. """
    def __init__(self,
                 dim,
                 img_size,
                 depth,
                 num_heads,
                 mlp_ratio=4,
                 drop_attn=0.,
                 drop_path=0.1,
                 mod_drop=0.):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.depth = depth
        self.blocks = nn.ModuleList([
            Cross_ViTransformerBlock_xyd(dim=dim,
                                     num_heads=num_heads,
                                     img_size=img_size,
                                     mlp_ratio=mlp_ratio,
                                     drop_attn=drop_attn,
                                     drop_path=drop_path,
                                     mod_drop=mod_drop)
                        for i in range(depth)])
        # final per-branch normalization before fusion
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)

    def forward(self, x, y, z):
        'x, y, z: [B, C, H, W], output: [B, C, H, W] x3'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        z = z.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y, z = blk(x, y, z)
        x_fus = x.transpose(1, 2).view(B, C, H, W)
        y_fus = y.transpose(1, 2).view(B, C, H, W)
        z_fus = z.transpose(1, 2).view(B, C, H, W)
        return x_fus, y_fus, z_fus


class JointCrossAttention(nn.Module):
    """ Joint cross-modal attention: the query branch attends to the concatenated
    tokens of the other two modalities within a single softmax (TokenFusion,
    CVPR 2022; CMNeXt hub2fuse, CVPR 2023; GeminiFusion, ICML 2024).
    Replaces two pairwise CrossAttention modules per branch.
    """
    def __init__(self, dim: int, num_heads: int,
                            img_size: int, attn_drop: float = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        assert dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        ## q from the main branch, shared k/v projection for both context modalities
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.output_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        ## learnable modality embeddings let the shared kv projection tell the two context modalities apart
        self.ctx_embed = nn.Parameter(torch.zeros(2, 1, 1, dim))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        self.register_buffer("relative_coords_table", _relative_coords_table(img_size), persistent=False)
        self.register_buffer("relative_position_index", _relative_position_index(img_size), persistent=False)

    def forward(self, x, ctx1, ctx2):
        '''x: [B, N, C] query branch; ctx1, ctx2: [B, N, C] the other two modalities'''
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nH, N, hd]
        ctx = torch.cat([ctx1 + self.ctx_embed[0], ctx2 + self.ctx_embed[1]], dim=1)  # [B, 2N, C]
        kv = self.kv_proj(ctx).reshape(B, 2 * N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, nH, 2N, hd]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nH, N, 2N]
        ## relative position bias (same spatial bias for both context modality segments)
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.img_size * self.img_size, self.img_size * self.img_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, N, N]
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + torch.cat([relative_position_bias, relative_position_bias], dim=-1).unsqueeze(0)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.output_proj(x)
        return x


class Joint_CrossViTBlock_xyd(nn.Module):
    """ Three-branch fusion block built on JointCrossAttention: one attention per
    branch over the concatenated tokens of the other two modalities, instead of
    six pairwise cross-attentions summed with softmax mixing weights.
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
        # joint cross attention (one per branch)
        self.attn_x = JointCrossAttention(dim, num_heads, img_size, drop_attn)
        self.attn_y = JointCrossAttention(dim, num_heads, img_size, drop_attn)
        self.attn_d = JointCrossAttention(dim, num_heads, img_size, drop_attn)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_x = nn.LayerNorm(dim)
        self.norm2_y = nn.LayerNorm(dim)
        self.norm2_d = nn.LayerNorm(dim)
        # MLP
        self.mlp_x = MLP(dim, mlp_ratio)
        self.mlp_y = MLP(dim, mlp_ratio)
        self.mlp_d = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor):
        '''x, y, d: [B, H*W, C]'''
        x_norm = self.norm1_x(x)
        y_norm = self.norm1_y(y)
        d_norm = self.norm1_d(d)
        # each branch attends jointly to the other two modalities
        x = x + self.drop_path(self.attn_x(x_norm, y_norm, d_norm))
        y = y + self.drop_path(self.attn_y(y_norm, x_norm, d_norm))
        d = d + self.drop_path(self.attn_d(d_norm, x_norm, y_norm))
        # FFN
        x = x + self.drop_path(self.mlp_x(self.norm2_x(x)))
        y = y + self.drop_path(self.mlp_y(self.norm2_y(y)))
        d = d + self.drop_path(self.mlp_d(self.norm2_d(d)))
        return x, y, d


class Joint_Vit_Fusion(nn.Module):
    """ Bottleneck fusion stage stacking Joint_CrossViTBlock_xyd blocks.
    Same interface as Cross_Vit_Fusion.
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
        self.blocks = nn.ModuleList([
            Joint_CrossViTBlock_xyd(dim=dim,
                                     num_heads=num_heads,
                                     img_size=img_size,
                                     mlp_ratio=mlp_ratio,
                                     drop_attn=drop_attn,
                                     drop_path=drop_path)
                        for i in range(depth)])
        # final per-branch normalization before fusion
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(dim)

    def forward(self, x, y, z):
        'x, y, z: [B, C, H, W], output: [B, C, H, W] x3'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        z = z.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y, z = blk(x, y, z)
        x = self.norm_x(x); y = self.norm_y(y); z = self.norm_z(z)
        x_fus = x.transpose(1, 2).view(B, C, H, W)
        y_fus = y.transpose(1, 2).view(B, C, H, W)
        z_fus = z.transpose(1, 2).view(B, C, H, W)
        return x_fus, y_fus, z_fus


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA / SW-MSA)"""
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

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        self.register_buffer("relative_coords_table", _relative_coords_table(window_size), persistent=False)
        self.register_buffer("relative_position_index", _relative_position_index(window_size), persistent=False)

    def forward(self, windows_x, mask):
        B, N, C = windows_x.shape
        qkv = self.qkv(windows_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Cross_WindowAttention(nn.Module):
    """Window-based Multi-head Cross-Attention (q from x, k/v from y)"""
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

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        self.register_buffer("relative_coords_table", _relative_coords_table(window_size), persistent=False)
        self.register_buffer("relative_position_index", _relative_position_index(window_size), persistent=False)

    def forward(self, windows_x: torch.Tensor, windows_y: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = windows_x.shape
        q_x = self.q(windows_x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv_y = self.kv(windows_y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_x, k_y, v_y = q_x[0], kv_y[0], kv_y[1]
        attn = (q_x @ k_y.transpose(-2, -1)) * self.scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(   # type: ignore
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        windows_x_att = (attn @ v_y).transpose(1, 2).reshape(B, N, C)
        windows_x_att = self.proj(windows_x_att)
        windows_x_att = self.proj_drop(windows_x_att)
        return windows_x_att


class Cross_SwinTransformerBlock_xyd(nn.Module):
    """Three-branch swin block: per-branch self-attn + two cross-attn paths,
    combined with learnable softmax weights (3 terms per branch)."""
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

        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_y = nn.LayerNorm(dim)
        self.norm1_d = nn.LayerNorm(dim)
        self.w_x = nn.Parameter(torch.zeros(3))
        self.w_y = nn.Parameter(torch.zeros(3))
        self.w_d = nn.Parameter(torch.zeros(3))

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

        self.mlp_x = MLP(dim, mlp_ratio)
        self.mlp_y = MLP(dim, mlp_ratio)
        self.mlp_d = MLP(dim, mlp_ratio)

        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def _create_mask(self, input_resolution: tuple[int, int]) -> torch.Tensor:
        H, W = input_resolution
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
                shift_mask[:, h, w, :] = cnt
                cnt += 1
        shift_mask_windows = window_partition(shift_mask, self.window_size)
        shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _windows(self, t: torch.Tensor) -> torch.Tensor:
        """[B, L, C] -> shifted window tokens [B*nW, ws*ws, C]"""
        H, W = self.input_resolution
        B, L, C = t.shape
        t = t.view(B, H, W, C)
        if self.shift_size > 0:
            t = torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        wins = window_partition(t, self.window_size)
        return wins.view(-1, self.window_size * self.window_size, C)

    def _merge(self, wins: torch.Tensor, C: int, B: int) -> torch.Tensor:
        """window tokens back to [B, L, C]"""
        H, W = self.input_resolution
        wins = wins.view(-1, self.window_size, self.window_size, C)
        t = window_reverse(wins, self.window_size, H, W)
        if self.shift_size > 0:
            t = torch.roll(t, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return t.view(B, H * W, C)

    def forward(self, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor):
        '''x, y, d: [B, H*W, C]'''
        B, L, C = x.shape
        shortcut_x, shortcut_y, shortcut_d = x, y, d
        x_norm = self.norm1_x(x)
        y_norm = self.norm1_y(y)
        d_norm = self.norm1_d(d)
        wx = torch.softmax(self.w_x, dim=0)
        wy = torch.softmax(self.w_y, dim=0)
        wd = torch.softmax(self.w_d, dim=0)

        win_x = self._windows(x_norm)
        win_y = self._windows(y_norm)
        win_d = self._windows(d_norm)

        x_att = wx[0] * self.window_attn_x_x(win_x, self.attn_mask) \
              + wx[1] * self.cross_window_attn_x_y(win_x, win_y, self.attn_mask) \
              + wx[2] * self.cross_window_attn_x_d(win_x, win_d, self.attn_mask)
        y_att = wy[0] * self.window_attn_y_y(win_y, self.attn_mask) \
              + wy[1] * self.cross_window_attn_y_x(win_y, win_x, self.attn_mask) \
              + wy[2] * self.cross_window_attn_y_d(win_y, win_d, self.attn_mask)
        d_att = wd[0] * self.window_attn_d_d(win_d, self.attn_mask) \
              + wd[1] * self.cross_window_attn_d_x(win_d, win_x, self.attn_mask) \
              + wd[2] * self.cross_window_attn_d_y(win_d, win_y, self.attn_mask)

        x = shortcut_x + self.drop_path(self._merge(x_att, C, B))
        y = shortcut_y + self.drop_path(self._merge(y_att, C, B))
        d = shortcut_d + self.drop_path(self._merge(d_att, C, B))

        x = x + self.drop_path(self.mlp_x(self.norm2_x(x)))
        y = y + self.drop_path(self.mlp_y(self.norm2_y(y)))
        d = d + self.drop_path(self.mlp_d(self.norm2_d(d)))
        return x, y, d


class Cross_SwinBasicLayer(nn.Module):
    """ A stack of three-branch cross-swin blocks for one stage. """
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
        'x, y, d: [B, C, H, W], output: [B, C, H, W] x3'
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        y = y.flatten(2).permute(0, 2, 1).contiguous()
        d = d.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.blocks:
            x, y, d = blk(x, y, d)
        x_fus = x.transpose(1, 2).view(B, C, H, W)
        y_fus = y.transpose(1, 2).view(B, C, H, W)
        d_fus = d.transpose(1, 2).view(B, C, H, W)
        return x_fus, y_fus, d_fus


class u3net_cross_fusion_v2(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0',
                        pretrained=True,
                        shallow_fusion=False,
                        mod_drop=0.,
                        bottleneck_fusion=True):
        '''
        shallow_fusion: enable windowed cross-modal fusion at skip layers 3/2.
        mod_drop: train-time modality dropout prob in the bottleneck fusion
            (0 = off; adds no parameters, inference unchanged).
        bottleneck_fusion: ablation switch — False removes the bottleneck
            cross-fusion entirely (encoder features pass straight to the
            decoder); everything else is identical.
        '''
        super().__init__()
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
        self.decode_channels = [96, 96, 96, 96, 48]   ##  from deep to shallow
        ## r4: reverted to pairwise Cross_Vit_Fusion (joint fusion regressed, see header)
        self.cross_layers_4 = Cross_Vit_Fusion(dim=self.out_channels[4],
                                            img_size=16,
                                            depth=4,
                                            num_heads=4,
                                            mlp_ratio=2,
                                            drop_attn=0.1,
                                            drop_path=0.2,
                                            mod_drop=mod_drop) if bottleneck_fusion else None
        self.shallow_fusion = shallow_fusion
        if shallow_fusion:
            self.cross_layers_3 = Cross_SwinBasicLayer(dim=self.out_channels[3],
                                                input_resolution=(32, 32),
                                                depth=2,
                                                num_heads=4,
                                                window_size=16,
                                                mlp_ratio=2,
                                                drop_attn=0.,
                                                drop_path=0.1)
            self.cross_layers_2 = Cross_SwinBasicLayer(dim=self.out_channels[2],
                                                input_resolution=(64, 64),
                                                depth=2,
                                                num_heads=4,
                                                window_size=16,
                                                mlp_ratio=2,
                                                drop_attn=0.,
                                                drop_path=0.1)

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
        self.aux2 = nn.Conv2d(self.out_channels[2]*3, 1, 3, padding=1)
        self.aux3 = nn.Conv2d(self.out_channels[3]*3, 1, 3, padding=1)
        self.aux4 = nn.Conv2d(self.out_channels[4]*3, 1, 3, padding=1)

    def forward(self, x, lat=None):       ## input size: 7x512x512
        x_opt = x[:, :3, :, :]
        x_nir = x[:, 3:6, :, :]
        x_dem = x[:, 6:, :, :]

        ## encoder part
        feas_opt = self.encoder_opt(x_opt)
        feas_nir = self.encoder_nir(x_nir)
        feas_dem = self.encoder_dem(x_dem)

        # layer 4
        fea_opt, fea_nir, fea_dem = feas_opt[-1], feas_nir[-1], feas_dem[-1]   # [B, C, 16, 16]
        if self.cross_layers_4 is not None:
            fea_opt, fea_nir, fea_dem = self.cross_layers_4(fea_opt, fea_nir, fea_dem)
        fea_cat_4 = torch.cat([fea_opt, fea_nir, fea_dem], dim=1)
        aux4_out = self.aux4(fea_cat_4)
        fea_fus_4 = self.DecoderBlocks[0](fea_cat_4)
        fea_fus_4 = self.up2(fea_fus_4)

        # layer 3
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-2], feas_nir[-2], feas_dem[-2]
        if self.shallow_fusion:
            skip_fea_opt, skip_fea_nir, skip_fea_dem = self.cross_layers_3(skip_fea_opt, skip_fea_nir, skip_fea_dem)
        fea_cat_3 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        aux3_out = self.aux3(fea_cat_3)
        fea_fus_3 = torch.cat([fea_fus_4, fea_cat_3], dim=1)
        fea_fus_3 = self.DecoderBlocks[1](fea_fus_3)
        fea_fus_3 = self.up2(fea_fus_3)

        # layer 2
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-3], feas_nir[-3], feas_dem[-3]
        if self.shallow_fusion:
            skip_fea_opt, skip_fea_nir, skip_fea_dem = self.cross_layers_2(skip_fea_opt, skip_fea_nir, skip_fea_dem)
        fea_cat_2 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        aux2_out = self.aux2(fea_cat_2)
        fea_fus_2 = torch.cat([fea_fus_3, fea_cat_2], dim=1)
        fea_fus_2 = self.DecoderBlocks[2](fea_fus_2)
        fea_fus_2 = self.up2(fea_fus_2)

        # layer 1
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-4], feas_nir[-4], feas_dem[-4]
        skip_fus_1 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        fea_fus_1 = torch.cat([fea_fus_2, skip_fus_1], dim=1)
        fea_fus_1 = self.DecoderBlocks[3](fea_fus_1)
        fea_fus_1 = self.up2(fea_fus_1)

        # layer 0
        skip_fea_opt, skip_fea_nir, skip_fea_dem = feas_opt[-5], feas_nir[-5], feas_dem[-5]
        fea_fus_0 = torch.cat([skip_fea_opt, skip_fea_nir, skip_fea_dem], dim=1)
        fea_fus_0 = torch.cat([fea_fus_1, fea_fus_0], dim=1)
        fea_fus_0 = self.DecoderBlocks[4](fea_fus_0)
        fea_fus_0 = self.up2(fea_fus_0)
        logit = self.outp(fea_fus_0)    # 1x512x512
        return logit, aux4_out, aux3_out, aux2_out


if __name__ == '__main__':
    model = u3net_cross_fusion_v2(backbone_name='efficientnet_b0',
                                  pretrained=False)
    x = torch.randn(2, 7, 512, 512)
    outs = model(x)
    for o in outs:
        print(o.shape)
