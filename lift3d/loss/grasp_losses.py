# lift3d/loss/grasp_losses.py
# Copyright (c) 2025 Lift3D authors.
# SPDX-License-Identifier: MIT
"""
Loss functions for TokenVoxelGraspActor
--------------------------------------
L_total = w_h * L_heatmap  +  w_q * L_quat  +  w_g * L_gripper
  • L_heatmap  : (Focal) BCE on coarse 3‑D grasp heatmap
  • L_quat     : symmetric quaternion distance
  • L_gripper  : BCE on open/close logits
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# (1) Heatmap loss
# --------------------------------------------------------------------------- #
def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal BCE adapted for logits input (no sigmoid applied).
    Works for dense [B,1,D,H,W] or sparse gathered tensors.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma) * (alpha * target + (1 - alpha) * (1 - target))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def bce_or_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if use_focal:
        return focal_bce_with_logits(
            logits, target, alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )
    return F.binary_cross_entropy_with_logits(logits, target)


# --------------------------------------------------------------------------- #
# (2) Quaternion symmetric loss
# --------------------------------------------------------------------------- #
def quat_loss(
    quat_pred: torch.Tensor, quat_gt: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Symmetric distance: L = 1 - |<q_pred, q_gt>|
    Both inputs must be (B,4) and unit‑normalized.
    """
    q_pred = F.normalize(quat_pred, p=2, dim=-1)
    q_gt = F.normalize(quat_gt, p=2, dim=-1)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))  # (B,)
    loss = 1.0 - dot
    return loss.mean() if reduction == "mean" else loss.sum()


# --------------------------------------------------------------------------- #
# (3) Gripper BCE loss
# --------------------------------------------------------------------------- #
def gripper_bce(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float | None = None
) -> torch.Tensor:
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=weight)


# --------------------------------------------------------------------------- #
# (4) Aggregated loss class
# --------------------------------------------------------------------------- #
class GraspLoss(nn.Module):
    """
    Wraps the three losses with configurable weights.
    Example
    -------
    >>> criterion = GraspLoss(w_heat=1.0, w_quat=0.5, w_grip=0.1, use_focal=True)
    >>> loss = criterion(out_dict, gt_heat, gt_quat, gt_grip)
    """

    def __init__(
        self,
        *,
        w_heat: float = 1.0,
        w_quat: float = 0.5,
        w_grip: float = 0.1,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight_grip: float | None = None,
    ):
        super().__init__()
        self.w_heat = w_heat
        self.w_quat = w_quat
        self.w_grip = w_grip
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight_grip = pos_weight_grip

    # ----------------------------------------------------------- #
    def forward(
        self,
        model_out: dict,
        heat_gt: torch.Tensor,    # (B,1,D,H,W)
        quat_gt: torch.Tensor,    # (B,4)
        grip_gt: torch.Tensor,    # (B,)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns total loss and a dict of individual components
        """
        l_heat = bce_or_focal(
            model_out["heatmap"], heat_gt,
            use_focal=self.use_focal,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
        )

        l_quat = quat_loss(model_out["quat"], quat_gt)
        l_grip = gripper_bce(
            model_out["gripper"], grip_gt, pos_weight=self.pos_weight_grip
        )

        total = self.w_heat * l_heat + self.w_quat * l_quat + self.w_grip * l_grip
        return total, {"heat": l_heat, "quat": l_quat, "grip": l_grip}

# --- 在 grasp_losses.py 末尾或合适位置 -----------------
def compute_loss(preds: dict, actions: torch.Tensor):
    """
    Wrapper expected by train_policy.py
    `actions` is RLBench 8‑DoF tensor:  x y z qx qy qz qw gripper
    """
    heat_gt   = preds["heatmap"].detach()*0  # <- 没有真 GT 时给零；或从 dataset 取
    quat_gt   = actions[:, 3:7]
    grip_gt   = actions[:, 7:8]
    criterion = GraspLoss()                  # 默认权重即可；也可放全局
    return criterion(preds, heat_gt, quat_gt, grip_gt)
