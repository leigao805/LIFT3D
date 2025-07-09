# lift3d/models/voxel_utils.py
# Copyright (c) 2025 Lift3D authors.
# SPDX-License-Identifier: MIT
"""
将 Lift3dCLIP 输出的 patch‑token (xyz, feat) 体素化，生成稀疏坐标 + 特征，
供 Sparse 3D 网络（MinkowskiEngine / spconv / 自定义 Sparse UNet）使用。

Typical usage
-------------
>>> from lift3d.models import voxel_utils as vu
>>> coords, feats = vu.tokens_to_sparse_voxel(
...     xyz, feat, voxel_size=0.01,
...     pc_range=[-0.5, -0.5, -0.1, 0.5, 0.5, 0.4])
>>> stensor = vu.to_me_tensor(coords, feats)       # 若安装了 MinkowskiEngine
"""
from __future__ import annotations

import torch
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
# Optional back‑ends
# --------------------------------------------------------------------------- #
try:
    import MinkowskiEngine as ME  # type: ignore
    _ME_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ME_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Helper : 量化 xyz → int voxel coords
# --------------------------------------------------------------------------- #
def _quantize(
    xyz: torch.Tensor,               # (B, K, 3), float32
    voxel_size: float,
    pc_range: List[float],
) -> torch.Tensor:
    """
    将 xyz (米) 量化成 [batch, z, y, x] 4‑维整型坐标。
    pc_range : [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    assert xyz.dim() == 3 and xyz.size(-1) == 3, "Expect (B,K,3)"
    xyz_min = torch.as_tensor(pc_range[:3], dtype=xyz.dtype, device=xyz.device)
    voxel = (xyz - xyz_min) / voxel_size
    voxel = voxel.floor().to(torch.int32)  # (B,K,3)
    return voxel


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def tokens_to_sparse_voxel(
    tokens_xyz: torch.Tensor,          # (B, K, 3)
    tokens_feat: torch.Tensor,         # (B, K, C)
    *,
    voxel_size: float = 0.01,
    pc_range: Optional[List[float]] = None,
    origin_shift: Optional[torch.Tensor] = None,
    add_batch_indices: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    体素化函数 —— 将 patch‑token 映射到稀疏体素坐标。

    Parameters
    ----------
    tokens_xyz : (B,K,3)
        3‑D 中心坐标，单位 **米**，与原点坐标系一致。
    tokens_feat : (B,K,C)
        patch‑token 特征；梯度将沿着归一化平均传播。
    voxel_size : float
        体素边长（米）。建议与抓取热图输出分辨率一致，如 1 cm。
    pc_range : [xmin, ymin, zmin, xmax, ymax, zmax]
        点云范围 (米)。若为 None，则自动按 xyz 最小/最大值向下/上取整 1 voxel。
    add_batch_indices : bool
        若 True，返回坐标 shape = (N, 4)，第一列是 batch_id；
        否则 (N, 3)。

    Returns
    -------
    coords : torch.IntTensor (N, 4) or (N, 3)
    feats  : torch.FloatTensor (N, C)
    """
    B, K, _ = tokens_xyz.shape

    # ---- ① 可选：先整体平移坐标系 ----
    if origin_shift is not None:
        # origin_shift shape 必须为 (3,)，且与 tokens_xyz 同 dtype / device
        tokens_xyz = tokens_xyz - origin_shift.to(tokens_xyz.device)[None, None, :]

    if pc_range is None:
        xyz_min = tokens_xyz.amin(dim=(0, 1))
        xyz_max = tokens_xyz.amax(dim=(0, 1))
        pc_range = [
            (xyz_min[0] // voxel_size) * voxel_size,
            (xyz_min[1] // voxel_size) * voxel_size,
            (xyz_min[2] // voxel_size) * voxel_size,
            (xyz_max[0] // voxel_size + 1) * voxel_size,
            (xyz_max[1] // voxel_size + 1) * voxel_size,
            (xyz_max[2] // voxel_size + 1) * voxel_size,
        ]

    voxel = _quantize(tokens_xyz, voxel_size, pc_range)        # (B,K,3)
    batch_idx = (
        torch.arange(B, device=tokens_xyz.device)
        .view(B, 1, 1)
        .expand_as(voxel[:, :, :1])
    )  # (B,K,1)

    if add_batch_indices:
        coords = torch.cat([batch_idx, voxel], dim=-1)         # (B,K,4)
    else:
        coords = voxel                                         # (B,K,3)

    coords = coords.reshape(-1, coords.shape[-1])              # → (N, 4/3)
    feats = tokens_feat.reshape(-1, tokens_feat.shape[-1])     # → (N, C)

    # 去重：同一体素取所有 token 特征平均
    # -------- 唯一键：batch 维放最高位，保证不同 batch 不混淆 --------
    if add_batch_indices:
        coords_hash = (
            coords[:, 0].to(torch.int64) << 48 |
            coords[:, 1].to(torch.int64) << 32 |
            coords[:, 2].to(torch.int64) << 16 |
            coords[:, 3].to(torch.int64)
        )
    else:
        coords_hash = (
            coords[:, 0].to(torch.int64) << 32 |
            coords[:, 1].to(torch.int64) << 16 |
            coords[:, 2].to(torch.int64)
        )

    uniq, inv = coords_hash.unique(return_inverse=True)
    feats_agg = torch.zeros(
        (uniq.numel(), feats.size(1)), dtype=feats.dtype, device=feats.device
    )
    feats_agg.index_add_(0, inv, feats)
    counts = torch.bincount(inv, minlength=uniq.numel()).unsqueeze(-1).clamp_min_(1)
    feats_agg = feats_agg / counts

    coords_out = coords.new_empty((uniq.numel(), coords.size(1)))
    coords_out[inv] = coords                                      # 重排回唯一顺序
    coords_out = coords_out.unique(dim=0)

    return coords_out.int(), feats_agg


# --------------------------------------------------------------------------- #
# Optional helper : convert to MinkowskiEngine Tensor
# --------------------------------------------------------------------------- #
def to_me_tensor(
    coords: torch.Tensor,
    feats: torch.Tensor,
    spatial_shape: list[int] | None = None
) -> "ME.SparseTensor":  # type: ignore
    """
    Wrap (coords, feats) to MinkowskiEngine SparseTensor.
    coords : int32 [N,4]  (batch,x,y,z)
    feats  : float32 [N,C]
    """
    # 统一走最稳妥路径：让 MinkowskiEngine 自己创建 / 管理 CoordinateManager
    # 0.5.x 版本会自动推断 spatial shape，无需手动指定。
    return ME.SparseTensor(feats, coords)