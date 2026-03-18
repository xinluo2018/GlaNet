import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────
# 基础卷积模块
# ─────────────────────────────────────────────────────────────────────

class ConvBnAct(nn.Sequential):
    """Conv2d -> BN -> Activation"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        act: nn.Module = None,
    ):
        act = act or nn.ReLU(inplace=True)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
        )


class DoubleConv(nn.Module):
    """
    经典 UNet 双卷积块：
    (Conv -> BN -> ReLU) x 2
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            ConvBnAct(in_ch, mid_ch),
            ConvBnAct(mid_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────
# Attention Gate（可选）
# ─────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention UNet 中的 Attention Gate。
    论文: "Attention U-Net: Learning Where to Look for the Pancreas"

    g: decoder 上采样特征(gate signal)
    x: encoder skip connection feature
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # psi: 1x1 Conv + Sigmoid 输出注意力权重
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 对齐空间尺寸
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear",
                               align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi   # 加权后的 skip feature


# ─────────────────────────────────────────────────────────────────────
# 单个 Decoder Block
# ─────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    UNet Decoder Block:
        1. 双线性上采样 x2(或转置卷积)
        2. [可选] Attention Gate 对 skip 加权
        3. Concat(upsampled, skip)
        4. DoubleConv
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_attention: bool = False,
        use_transpose_conv: bool = False,
    ):
        super().__init__()

        # 上采样方式
        if use_transpose_conv:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch // 2, kernel_size=2, stride=2
            )
            up_out_ch = in_ch // 2
        else:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            up_out_ch = in_ch

        # Attention Gate
        self.attention = None
        if use_attention and skip_ch > 0:
            self.attention = AttentionGate(
                g_ch=up_out_ch,
                x_ch=skip_ch,
                inter_ch=skip_ch // 2,
            )

        # 融合后的双卷积
        self.conv = DoubleConv(up_out_ch + skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            # 防止奇数输入造成的尺寸不对齐
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=True)
            # 可选 Attention Gate
            if self.attention is not None:
                skip = self.attention(g=x, x=skip)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────
# UNet Decoder（完整解码器）
# ─────────────────────────────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """
    通用 UNet Decoder。

    接收 encoder_channels（从浅到深），
    从最深层开始逐步上采样并融合 skip connection。

    Args:
        encoder_channels : 编码器各阶段通道数，从浅到深
                           e.g. [64, 64, 128, 256, 512] (ResNet34)
        decoder_channels : 解码器各 block 的输出通道数，从深到浅
                           e.g. [256, 128, 64, 32, 16]
        use_attention    : 是否启用 Attention Gate
        use_transpose_conv: 是否使用转置卷积代替双线性上采样
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        use_attention: bool = False,
        use_transpose_conv: bool = False,
    ):
        super().__init__()

        # skip channels: 除最深层外，反转后从次深到最浅
        # e.g. encoder=[64,64,128,256,512] -> skips=[256,128,64,64]
        skip_channels = list(reversed(encoder_channels[:-1]))

        # decoder_channels 数量可以超过 skip 层数（此时无 skip）
        blocks = []
        in_ch = encoder_channels[-1]

        for i, out_ch in enumerate(decoder_channels):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            blocks.append(
                DecoderBlock(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    use_attention=use_attention,
                    use_transpose_conv=use_transpose_conv,
                )
            )
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = decoder_channels[-1]

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: encoder 输出特征列表，从浅到深
                      e.g. [f0, f1, f2, f3, f4]
        Returns:
            decoded feature map
        """
        # 从最深特征开始
        x = features[-1]
        # skip connections: 次深到最浅
        skips = list(reversed(features[:-1]))

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)

        return x
