# model/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .encoder import TimmEncoder
from .decoder import UNetDecoder


class SegmentationHead(nn.Module):
    """分割输出头: 1x1 卷积 + 可选的上采样"""
    def __init__(self, in_ch: int, num_classes: int,
                 scale_factor: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode="bilinear", align_corners=True)
        return x


class TimmUNet(nn.Module):
    """
    UNet with Timm Backbone Encoder.

    ┌──────────────────────────────────────────┐
    │  Input Image  [B, C, H, W]               │
    │       |                                  │
    │  TimmEncoder (features_only=True)        │
    │  └── [f0, f1, f2, f3, f4]               │
    │              |                           │
    │  UNetDecoder (skip connections)          │
    │  └── decoded feature map                 │
    │              |                           │
    │  SegmentationHead (1x1 Conv)             │
    │  └── [B, num_classes, H, W]              │
    └──────────────────────────────────────────┘

    Args:
        backbone_name    : timm 模型名称
        num_classes      : 分割输出类别数
        pretrained       : 是否使用预训练权重
        in_chans         : 输入图像通道数
        decoder_channels : 解码器各阶段输出通道数（从深到浅）
        use_attention    : 是否启用 Attention Gate (Attention UNet)
        use_transpose_conv: 是否使用转置卷积上采样
        freeze_stages    : 冻结前 N 个编码器 stage（-1 不冻结）
        out_indices      : 使用 encoder 的哪几个 stage
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 1,
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        use_attention: bool = False,
        use_transpose_conv: bool = False,
        freeze_stages: int = -1,
        out_indices: tuple = (0, 1, 2, 3, 4),
    ):
        super().__init__()

        # ── 1. Encoder ────────────────────────────────────────────────
        self.encoder = TimmEncoder(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            out_indices=out_indices,
            freeze_stages=freeze_stages,
        )

        print(f"[TimmUNet] Backbone : {backbone_name}")
        print(f"[TimmUNet] Enc channels : {self.encoder.out_channels}")
        print(f"[TimmUNet] Reductions   : {self.encoder.reductions}")

        # ── 2. Decoder ────────────────────────────────────────────────
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(decoder_channels),
            use_attention=use_attention,
            use_transpose_conv=use_transpose_conv,
        )

        # ── 3. Segmentation Head ──────────────────────────────────────
        # 计算 decoder 输出相对原图的下采样倍率
        # 通常最浅 skip 的 reduction 就是 decoder 输出的 scale
        first_reduction = self.encoder.reductions[0]   # 通常是 2
        scale_factor = float(first_reduction)

        self.head = SegmentationHead(
            in_ch=self.decoder.out_channels,
            num_classes=num_classes,
            scale_factor=scale_factor,   # 补齐到原图尺寸
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            logits: [B, num_classes, H, W]，与输入同尺寸
        """
        input_size = x.shape[-2:]

        # Step 1: 多尺度特征提取
        features = self.encoder(x)

        # Step 2: 解码（含 skip connections）
        decoded = self.decoder(features)

        # Step 3: 输出头
        logits = self.head(decoded)

        # 安全兜底：确保输出与输入完全同尺寸
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size,
                mode="bilinear", align_corners=True
            )
        return logits

    def get_param_groups(self, lr_backbone: float, lr_decoder: float):
        """
        为 backbone 和 decoder 设置不同学习率（常用技巧）
        
        Usage:
            optimizer = AdamW(model.get_param_groups(1e-4, 1e-3))
        """
        return [
            {"params": self.encoder.parameters(), "lr": lr_backbone},
            {"params": self.decoder.parameters(), "lr": lr_decoder},
            {"params": self.head.parameters(),    "lr": lr_decoder},
        ]


# ─────────────────────────────────────────────────────────────────────
# 工厂函数：快速构建常见配置
# ─────────────────────────────────────────────────────────────────────

def build_unet(
    backbone: str = "resnet34",
    num_classes: int = 1,
    pretrained: bool = True,
    attention: bool = False,
    **kwargs
) -> TimmUNet:
    """一行代码构建 TimmUNet"""
    return TimmUNet(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        use_attention=attention,
        **kwargs
    )
