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

def pose_to_heatmap(xyz_gt: torch.Tensor,
                    xyz_min: torch.Tensor,  # (B,3)  or broadcastable
                    voxel_size: torch.Tensor,
                    grid_size: torch.Tensor):
    """
    xyz_gt  : (B,3)   in metres, same world frame as pc_range
    xyz_min : (B,3)   per‑batch origin (the same one the Actor subtracted)
    voxel_size: scalar tensor
    grid_size : (3,)   [D,H,W]
    """
    B = xyz_gt.shape[0]
    D, H, W = grid_size.tolist()
    idx = ((xyz_gt - xyz_min) / voxel_size).long()          # (B,3)
    heat = torch.zeros(B, 1, D, H, W, device=xyz_gt.device)
    valid = ((idx[:, 0] >= 0) & (idx[:, 0] < W) &
             (idx[:, 1] >= 0) & (idx[:, 1] < H) &
             (idx[:, 2] >= 0) & (idx[:, 2] < D))
    if valid.any():
        b_ids = torch.arange(B, device=xyz_gt.device)[valid]
        heat[b_ids, 0, idx[valid, 2], idx[valid, 1], idx[valid, 0]] = 1.0
    return heat


# lift3d/loss/grasp_losses.py
# --------------------------------------------
def compute_loss(preds, actions: torch.Tensor):
    """
    既兼容训练产生的 `dict`，也兼容验证阶段得到的 (B,8) action tensor。
    `actions` 总是 RLBench 8‑DoF tensor：x y z qx qy qz qw gripper
    """
    # ---------- 1) 如果前向返回 dict（训练阶段） ----------
    if isinstance(preds, dict):
        xyz_gt  = actions[:, :3]
        quat_gt = actions[:, 3:7]
        grip_gt = actions[:, 7:8]
        # =========================================================
        # >>>>>>> 【就在这里插入临时调试代码】 <<<<<<<<
        #
        # ---------- 0) 调试：检查 GT 是否落在体素网格内 ----------
        with torch.no_grad():           # 避免干扰梯度
            voxel_size = preds["voxel_size"]      # scalar tensor
            grid_size  = preds["grid_size"]       # (3,)
            xyz_min    = preds["xyz_min"]         # (B,3)

            # 反向找出 one‑hot 为 1 的体素整数坐标 idx
            idx = ((xyz_gt - xyz_min) / voxel_size).long()
            D, H, W = grid_size.tolist()
            oob = (
                (idx[:, 0] < 0) | (idx[:, 0] >= W) |
                (idx[:, 1] < 0) | (idx[:, 1] >= H) |
                (idx[:, 2] < 0) | (idx[:, 2] >= D)
            )
            # 只在第一次 step 打印即可；或用 logger
            if torch.rand(1).item() < 0.01:       # 随机抽 1% step
                print(f">> OOB ratio: {oob.float().mean().item():.3f}")

            # 还原到世界坐标的体素中心
            center_xyz = preds["xyz_min"] + (idx.float() + 0.5) * preds["voxel_size"]
            err = (center_xyz - xyz_gt).norm(dim=-1)   # (B,)
            print("mean EE‑voxel‑center err (m):", err.mean().item())


        # =========================================================

        heat_gt = pose_to_heatmap(
            xyz_gt,
            preds["xyz_min"],
            preds["voxel_size"],
            preds["grid_size"]
        )

        heat_sum = heat_gt.flatten(1).sum(dim=1)         # (B,)
        assert (heat_sum == 1).all(), \
            "GT heatmap 非 one‑hot！请检查 pose_to_heatmap() 与坐标系是否匹配"

        criterion = GraspLoss()
        return criterion(preds, heat_gt, quat_gt, grip_gt)

    # ---------- 2) 如果前向返回 (B,8) tensor（验证阶段） ----------
    elif torch.is_tensor(preds) and preds.ndim == 2 and preds.size(1) == 8:
        quat_pred = preds[:, 3:7]
        quat_gt   = actions[:, 3:7]

        grip_pred = preds[:, 7:8]
        grip_gt   = actions[:, 7:8]

        # 2‑a) quaternion 对齐损失
        l_quat = quat_loss(quat_pred, quat_gt)

        # 2‑b) gripper BCE
        l_grip = gripper_bce(grip_pred, grip_gt)

        total = l_quat + l_grip      # heatmap_loss 置 0
        return total, {"heat": torch.tensor(0., device=preds.device),
                       "quat": l_quat, "grip": l_grip}

    else:
        raise TypeError(f"`preds` must be dict or (B,8) tensor, got {type(preds)}")