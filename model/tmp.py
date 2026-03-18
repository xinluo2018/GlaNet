import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional


# ─────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """Conv2d + BN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    单个 Decoder Block:
      1. 双线性上采样 x2
      2. Concat skip connection
      3. 两次 ConvBnRelu
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        # skip_ch == 0 表示该层没有 skip connection (bottleneck 层)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # 防止尺寸不匹配（奇数输入）
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# 编码器：Timm Backbone Wrapper
# ─────────────────────────────────────────────

class TimmEncoder(nn.Module):
    """
    封装任意 timm 模型为多尺度特征提取器。
    
    利用 timm 的 `features_only=True` 接口，
    自动获取每个 stage 的输出通道数。
    """
    def __init__(self, backbone_name: str = "resnet34",
                 pretrained: bool = True,
                 in_chans: int = 3,
                 out_indices: tuple = (0, 1, 2, 3, 4)):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=out_indices,
        )
        # 每个 stage 输出通道数，e.g. [64, 64, 128, 256, 512]
        self.out_channels: List[int] = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


# ─────────────────────────────────────────────
# 主模型：TimmUNet
# ─────────────────────────────────────────────

class TimmUNet(nn.Module):
    """
    UNet with Timm Backbone Encoder.

    Args:
        backbone_name: timm 模型名，需支持 features_only=True
        num_classes:   输出通道数（分割类别数）
        pretrained:    是否加载预训练权重
        in_chans:      输入通道数（默认 3）
        decoder_channels: 每个 decoder block 的输出通道数（从深到浅）
    """
    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 1,
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────
        self.encoder = TimmEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        enc_channels = self.encoder.out_channels  # e.g. [64, 64, 128, 256, 512]

        # ── Decoder ──────────────────────────────
        # enc_channels 从浅到深排列，decoder 从深到浅
        # decoder[i] 接收 enc_channels[-(i+1)] 作为 skip
        decoder_blocks = []
        in_ch = enc_channels[-1]  # bottleneck 输出通道
        skip_channels = list(reversed(enc_channels[:-1]))  # 浅层 skip

        for i, out_ch in enumerate(decoder_channels):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.decoder = nn.ModuleList(decoder_blocks)

        # ── Segmentation Head ────────────────────
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录原始尺寸，最终恢复
        input_size = x.shape[-2:]

        # Encoder：获取多尺度特征列表
        features = self.encoder(x)
        # features: [f0(大), f1, f2, f3, f4(小)]

        # Decoder：从最深特征开始，逐步上采样
        x_dec = features[-1]
        skips = list(reversed(features[:-1]))  # 从次深到最浅

        for i, block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            x_dec = block(x_dec, skip)

        # 确保输出与输入同尺寸
        if x_dec.shape[-2:] != input_size:
            x_dec = F.interpolate(x_dec, size=input_size,
                                  mode="bilinear", align_corners=True)

        return self.head(x_dec)


# ─────────────────────────────────────────────
# 工厂函数：快速创建常用配置
# ─────────────────────────────────────────────

def build_timm_unet(
    backbone: str = "resnet34",
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs
) -> TimmUNet:
    return TimmUNet(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
