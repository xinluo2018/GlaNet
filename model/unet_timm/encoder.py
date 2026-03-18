# model/encoder.py

import torch
import torch.nn as nn
import timm
from typing import List, Tuple


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet34",
        pretrained: bool = True,
        in_chans: int = 3,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
        freeze_stages: int = -1,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices
        # ── 核心：features_only=True 模式 ────────────────────────────
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=out_indices,
        )

        # ── 自动读取 feature 元信息 ───────────────────────────────────
        # e.g. resnet34 -> channels=[64, 64, 128, 256, 512]
        #                  reductions=[2,  4,  8,  16,  32]
        self._out_channels: List[int] = self.backbone.feature_info.channels()
        self._reductions: List[int] = self.backbone.feature_info.reduction()

        # ── 可选：冻结若干 stage ───────────────────────────────────────
        if freeze_stages >= 0:
            self._freeze(freeze_stages)

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            features: List[Tensor], 长度 == len(out_indices)
                      按 stage 从浅到深排列
                      e.g. [f0, f1, f2, f3, f4]
                           shapes (B, C_i, H_i, W_i)
        """
        return self.backbone(x)

    # ─────────────────────────────────────────────────────────────────
    # 属性
    # ─────────────────────────────────────────────────────────────────

    @property
    def out_channels(self) -> List[int]:
        """每个 stage 输出的通道数，从浅到深"""
        return self._out_channels

    @property
    def reductions(self) -> List[int]:
        """每个 stage 相对于原图的下采样倍率"""
        return self._reductions

    @property
    def num_stages(self) -> int:
        return len(self.out_indices)

    # ─────────────────────────────────────────────────────────────────
    # 冻结参数
    # ─────────────────────────────────────────────────────────────────

    def _freeze(self, num_stages: int) -> None:
        """冻结前 num_stages 个 stage 的参数（节省显存，加快训练）"""
        for i, block in enumerate(self.backbone.children()):
            if i >= num_stages:
                break
            for param in block.parameters():
                param.requires_grad = False
        print(f"[TimmEncoder] Froze first {num_stages} stages of {self.model_name}")
