# lift3d/models/sparse_unet.py
# Copyright (c) 2025 Lift3D authors.
# SPDX-License-Identifier: MIT
"""
Sparse 3D U‑Net backbone for Lift3D grasp‑heatmap generation.

* in_channels  : 768  (Lift3dCLIP patch‑token dim)
* out_channels : 1    (coarse grasp heatmap)
* structure    : U‑Net (depth=3) with equal #channels per level doubled
 
Usage
-----
>>> import torch
>>> import MinkowskiEngine as ME
>>> from lift3d.models.sparse_unet import Sparse3DUNet
>>> coords = torch.randint(0, 64, (20000, 4), dtype=torch.int32)  # (N,4)
>>> feats  = torch.randn(20000, 768)
>>> stensor = ME.SparseTensor(feats, coords)
>>> net = Sparse3DUNet(in_ch=768, base_ch=96, depth=3)
>>> out = net(stensor)           # out: SparseTensor (N', 1)
"""

from __future__ import annotations
from typing import Tuple, List

import MinkowskiEngine as ME
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Helper blocks
# --------------------------------------------------------------------------- #
def conv_block(
    in_ch: int, out_ch: int, kernel_size: int = 3, dimension: int = 3
) -> nn.Module:
    return nn.Sequential(
        ME.MinkowskiConvolution(
            in_ch, out_ch, kernel_size=kernel_size, stride=1,
            dimension=dimension, bias=False
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiReLU(inplace=True),
    )


def downsample_block(ch: int, dimension: int = 3) -> ME.MinkowskiConvolution:
    return ME.MinkowskiConvolution(
        ch, ch, kernel_size=2, stride=2, dimension=dimension, bias=False
    )


def upsample_block(ch: int, dimension: int = 3) -> ME.MinkowskiConvolutionTranspose:
    return ME.MinkowskiConvolutionTranspose(
        ch, ch // 2, kernel_size=2, stride=2, dimension=dimension, bias=False
    )


# --------------------------------------------------------------------------- #
# Sparse 3D U‑Net
# --------------------------------------------------------------------------- #
class Sparse3DUNet(nn.Module):
    """
    depth=3 架构示意
    encoder:  in → E0 ─ds→ E1 ─ds→ E2
    decoder:  U2 ⇢ (+E1) → U1 ⇢ (+E0) → U0 → out
    """

    def __init__(
        self,
        in_ch: int = 768,
        base_ch: int = 96,
        depth: int = 3,
        out_ch: int = 1,
        dimension: int = 3,
    ):
        super().__init__()
        assert depth >= 2, "depth 必须 ≥2"

        enc_blocks: List[nn.Module] = []
        dec_blocks: List[nn.Module] = []
        ds_layers: List[nn.Module] = []
        us_layers: List[nn.Module] = []

        ch = base_ch
        # stem
        self.stem = conv_block(in_ch, ch, dimension=dimension)

        # encoder
        for d in range(1, depth):
            ds_layers.append(downsample_block(ch, dimension))
            enc_blocks.append(conv_block(ch, ch * 2, dimension=dimension))
            ch *= 2  # double channels per level

        # bottleneck
        self.bottleneck = conv_block(ch, ch, dimension=dimension)

        # decoder (inverse order)
        for d in range(depth - 1):
            us_layers.append(upsample_block(ch, dimension))
            ch //= 2
            dec_blocks.append(
                nn.Sequential(
                    conv_block(ch * 2, ch, dimension=dimension),
                    conv_block(ch, ch, dimension=dimension),
                )
            )

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.downsamples = nn.ModuleList(ds_layers)
        self.upsamples = nn.ModuleList(us_layers)

        # final 1×1 conv to heatmap
        self.head = ME.MinkowskiConvolution(
            base_ch, out_ch, kernel_size=1, bias=True, dimension=dimension
        )

    # --------------------------------------------------------------------- #
    def forward(self, x: "ME.SparseTensor") -> "ME.SparseTensor":  # type: ignore
        skip: List[ME.SparseTensor] = []

        # Stem + Encoder
        x = self.stem(x)
        for ds, enc in zip(self.downsamples, self.enc_blocks):
            skip.append(x)
            x = ds(x)
            x = enc(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for us, dec, sk in zip(self.upsamples, self.dec_blocks, reversed(skip)):
            x = us(x)
            x = ME.cat(x, sk)  # skip connection concat
            x = dec(x)

        # Head
        x = self.head(x)  # (N', out_ch)
        return x

    # ------------------------------------------------------------------ #
    @property
    def out_channels(self) -> int:
        return self.head.out_channels
