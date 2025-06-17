# lift3d/models/grasp_token_head.py
# Copyright (c) 2025 Lift3D authors.
# SPDX-License-Identifier: MIT
"""
Light‑weight MLP heads that map a single aggregated token feature
to (i) grasp orientation (unit quaternion) and (ii) gripper open‑state
probability.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _weights_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# (1) Orientation head : token -> quaternion
# --------------------------------------------------------------------------- #
class GraspOrientHead(nn.Module):
    """
    将 token 特征 (B,C) 回归到单位四元数 (B,4)。
    网络结构：C -> 256 -> 128 -> 4，然后做 L2‑normalize。

    Loss 推荐：
    ---------
        q_pred  : (B,4), q_gt : (B,4)
        loss = 1 - |⟨q_pred , q_gt⟩|
    """

    def __init__(self, in_dim: int = 768, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Linear(last, 4))
        self.mlp = nn.Sequential(*layers)
        self.apply(_weights_init)

    # ------------------------------------------------------------------ #
    def forward(self, token_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_feat : (B, C)

        Returns
        -------
        quat : (B, 4)  已归一化到单位四元数
        """
        quat = self.mlp(token_feat)           # (B,4)
        quat = F.normalize(quat, p=2, dim=-1) # L2 归一化
        return quat


# --------------------------------------------------------------------------- #
# (2) Gripper‑state head : token -> open/close probability
# --------------------------------------------------------------------------- #
class GripperStateHead(nn.Module):
    """
    输出夹爪开闭概率（未 Sigmoid 激活）。
    网络结构：C -> 256 -> 1
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.apply(_weights_init)

    # ------------------------------------------------------------------ #
    def forward(self, token_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_feat : (B, C)

        Returns
        -------
        logits : (B, 1)   —— 需在 loss 前 or 推理时做 sigmoid
        """
        x = F.relu(self.fc1(token_feat))
        logits = self.fc2(x)
        return logits
