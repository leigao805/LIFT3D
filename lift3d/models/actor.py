from __future__ import annotations

# lift3d/models/actor.py  (新增部分类)
# ---------------------------------------------------------------

import abc
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn

from lift3d.helpers.graphics import PointCloud
from lift3d.models.mlp.batchnorm_mlp import BatchNormMLP
from lift3d.models.mlp.mlp import MLP


import torch.nn.functional as F
from typing import Dict, Tuple

from lift3d.models.voxel_utils import tokens_to_sparse_voxel, to_me_tensor
from lift3d.models.sparse_unet import Sparse3DUNet
from lift3d.models.grasp_token_head import GraspOrientHead, GripperStateHead

class Actor(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, images, point_clouds, robot_states):
        pass


class VisionGuidedMLP(Actor):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_dropout_rate: float,
        robot_state_dim: int,
        robot_state_dropout_rate: float,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(VisionGuidedMLP, self).__init__()
        self.image_encoder = image_encoder
        self.image_dropout = nn.Dropout(image_dropout_rate)
        self.robot_state_encoder = nn.Linear(robot_state_dim, image_encoder.feature_dim)
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)
        self.policy_head = MLP(
            input_dim=2 * image_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )

    def forward(self, images, point_clouds, robot_states, texts):
        image_emb = self.image_encoder(images)
        image_emb = self.image_dropout(image_emb)
        robot_state_emb = self.robot_state_encoder(robot_states)
        robot_state_emb = self.robot_state_dropout(robot_state_emb)
        emb = torch.cat([image_emb, robot_state_emb], dim=1)
        actions = self.policy_head(emb)
        return actions


class PointCloudGuidedMLP(Actor):

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        point_cloud_dropout_rate: float,
        robot_state_dim: int,
        robot_state_dropout_rate: float,
        action_dim: int,
        policy_hidden_dims: List[int],
        policy_head_init_method: str,
    ):
        super(PointCloudGuidedMLP, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.point_cloud_dropout = nn.Dropout(point_cloud_dropout_rate)
        self.robot_state_encoder = nn.Linear(
            robot_state_dim, point_cloud_encoder.feature_dim
        )
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)
        self.policy_head = MLP(
            input_dim=2 * point_cloud_encoder.feature_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            init_method=policy_head_init_method,
        )

    def forward(self, images, point_clouds, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        point_cloud_emb = self.point_cloud_dropout(point_cloud_emb)
        robot_state_emb = self.robot_state_encoder(robot_states)
        robot_state_emb = self.robot_state_dropout(robot_state_emb)
        emb = torch.cat([point_cloud_emb, robot_state_emb], dim=1)
        actions = self.policy_head(emb)
        return actions


class VisionGuidedBatchNormMLP(Actor):
    def __init__(
        self,
        image_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        nonlinearity: str,
        dropout_rate: float,
    ):
        super(VisionGuidedBatchNormMLP, self).__init__()
        self.image_encoder = image_encoder
        self.policy_head = BatchNormMLP(
            input_dim=image_encoder.feature_dim + robot_state_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
        )
        for param in list(self.policy_head.parameters())[-2:]:
            param.data = 1e-2 * param.data

    def forward(self, images, point_clouds, robot_states, texts):
        image_emb = self.image_encoder(images)
        emb = torch.cat([image_emb, robot_states], dim=1)
        actions = self.policy_head(emb)
        return actions

class PointCloudGuidedBatchNormMLP(Actor):

    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        robot_state_dim: int,
        action_dim: int,
        policy_hidden_dims: List[int],
        nonlinearity: str,
        dropout_rate: float,
    ):
        super(PointCloudGuidedBatchNormMLP, self).__init__()
        self.point_cloud_encoder = point_cloud_encoder
        self.policy_head = BatchNormMLP(
            input_dim=point_cloud_encoder.feature_dim + robot_state_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=action_dim,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
        )
        for param in list(self.policy_head.parameters())[-2:]:
            param.data = 1e-2 * param.data

    def forward(self, images, point_clouds, robot_states, texts):
        # * Notice: normalize the input point cloud
        point_clouds = PointCloud.normalize(point_clouds)
        point_cloud_emb = self.point_cloud_encoder(point_clouds)
        emb = torch.cat([point_cloud_emb, robot_states], dim=1)
        actions = self.policy_head(emb)
        return actions

class TokenVoxelGraspActor(Actor):
    def __init__(
        self,
        point_cloud_encoder: nn.Module,
        *,
        robot_state_dim: int,
        action_dim: int | None = None,      # 兼容其它 Actor 的签名
        token_dropout_rate: float = 0.15,
        robot_state_dropout_rate: float = 0.10,
        voxel_size: float = 0.01,
        pc_range: Tuple[float, float, float, float, float, float] = (
            -0.3, -0.5, 0.0, 0.7, 0.5, 1.0
        ),
        k_nearest: int = 4,
        sparse_unet_cfg: Dict | None = None,
        orient_head_cfg: Dict | None = None,
        gripper_head_cfg: Dict | None = None,
        **kwargs,                            # 捕获将来可能的多余参数
    ):
        super().__init__()

        self.feat_dim = point_cloud_encoder.feature_dim          # e.g. 768
        fused_dim = self.feat_dim * 2                            # 1536

        # 1) encoders
        self.point_cloud_encoder = point_cloud_encoder
        self.token_dropout = nn.Dropout(token_dropout_rate)

        # 2) robot‑state branch
        self.robot_state_encoder = nn.Linear(robot_state_dim, self.feat_dim)
        self.robot_state_dropout = nn.Dropout(robot_state_dropout_rate)

        # 3) Sparse‑UNet —— in_channels 设为 2 × feat_dim
        su_cfg = dict(sparse_unet_cfg or {})
        su_cfg["in_ch"] = fused_dim
        self.sparse_unet = Sparse3DUNet(**su_cfg)

        # 4) heads（输入 1536）
        oh_cfg = {**(orient_head_cfg or {}), "in_dim": fused_dim}
        gh_cfg = {**(gripper_head_cfg or {}), "in_dim": fused_dim}
        self.orient_head = GraspOrientHead(**oh_cfg)
        self.gripper_head = GripperStateHead(**gh_cfg)

        # 5) geo‑hyper‑params
        self.voxel_size = voxel_size
        self.pc_range = list(pc_range)
        self.k_nearest = k_nearest

    def _dict2action(self, out_dict: dict) -> torch.Tensor:
        pos  = out_dict["voxel_center"]                    # (B,3)
        quat = F.normalize(out_dict["quat"], p=2, dim=-1)  # <- 保证单位四元数
        grip = torch.sigmoid(out_dict["gripper"])          # (B,1)
        return torch.cat([pos, quat, grip], dim=-1)        # (B,8)

    # ------------------------------------------------------------------
    def forward(
        self,
        images,                               # placeholder to keep API
        point_clouds: torch.Tensor,           # (B, N, 3)
        robot_states: torch.Tensor,           # (B, robot_state_dim)
        texts,                                # placeholder
    ):
        # 1. point‑cloud → tokens
        point_clouds = PointCloud.normalize(point_clouds)
        cls_tok, patch_tok, patch_xyz = self.point_cloud_encoder(
            point_clouds, return_tokens=True, return_xyz=True
        )  # (B,768) (B,K,768) (B,K,3)

        # 2. tokens → sparse voxels（用 helpers.voxel_utils 封装的通用函数）
        # ------------------------------------------------------------------
        #   ① 先“粗”体素化：得到稀疏坐标 & token 特征平均。
        shift = torch.tensor([0.0, 0.0, 0.6], device=patch_xyz.device)  # 例：把 z 轴整体减 0.6
        coords, feats, coord_offset = tokens_to_sparse_voxel(
            tokens_xyz=patch_xyz,           # (B, K, 3)
            tokens_feat=patch_tok,          # (B, K, 768)
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
            origin_shift=shift,             # ← 真正把平移量用上
            add_batch_indices=True,
            return_offset=True,        # ← 必须显式打开！
        )                                   # → coords (N,4), feats (N,768)

        # 3. robot state → embed & broadcast → concat
        rs_emb = self.robot_state_dropout(self.robot_state_encoder(robot_states))  # (B,768)
        rs_broadcast = rs_emb[coords[:, 0].long()]                                  # (N_total,768)
        feats = torch.cat([feats, rs_broadcast], dim=1)                             # (N_total,1536)

        # 4. sparse UNet → heatmap
        # ► 送入 ME 之前必须保证 xyz ≥0
        coords_me = coords.clone()
        if (coord_offset != 0).any():              # (3,)
            coords_me[:, 1:] -= coord_offset       # 只改 xyz

        me_tensor = to_me_tensor(coords_me, feats)
        heat_out  = self.sparse_unet(me_tensor).dense()

        # 如果 UNet 返回 (tensor, extra) 形式，取第一个分量
        if isinstance(heat_out, (tuple, list)):
            heat_dense = heat_out[0]
        else:
            heat_dense = heat_out                    # 原生 MinkowskiTensor

        # 5. choose voxel with max score
        B, _, D, H, W = heat_dense.shape
        flat_idx = heat_dense.view(B, -1).argmax(dim=1)
        z = (flat_idx // (H * W)).int()
        y = ((flat_idx % (H * W)) // W).int()
        x = (flat_idx % W).int()
        voxel_idx = torch.stack([x, y, z], dim=1).float()        # (B,3)  在局部网格坐标系

        # ► 还原到世界坐标：先加回 offset，再加上 pc_range.min 与 shift
        local_min = torch.tensor(self.pc_range[:3],
                                 device=heat_dense.device)       # (-0.3,-0.5,0)
        voxel_centers = (
            local_min                                     +
            (voxel_idx + 0.5) * self.voxel_size           +
            shift                                         +
            coord_offset.float() * self.voxel_size        # <‑‑ NEW
        )


        # 6. aggregate k‑NN patch tokens
        dist = torch.cdist(voxel_centers.unsqueeze(1), patch_xyz)                   # (B,1,K)
        idx = dist.topk(self.k_nearest, largest=False).indices                      # (B,1,k)
        idx = idx.expand(-1, patch_tok.size(-1), -1).transpose(1, 2)                # (B,k,768)
        token_agg = patch_tok.gather(dim=1, index=idx).mean(dim=1)                  # (B,768)
        token_agg = self.token_dropout(token_agg)

        # 7. robot state embed (same as above but per‑batch) – reuse rs_emb
        emb = torch.cat([token_agg, rs_emb], dim=1)                                 # (B,1536)

        # 8. heads
        quat_pred   = self.orient_head(emb)        # (B,4)
        gripper_log = self.gripper_head(emb)       # (B,1)

        # ----------------------------------------------------------
        # ☆ 把体素化过程的关键信息一并带回，以便 loss 能反推 heat‑GT
        # ----------------------------------------------------------

        xyz_min_world = (
            local_min + shift + coord_offset.float() * self.voxel_size
        ).unsqueeze(0).expand(point_clouds.size(0), 3)

        output = {
            "heatmap":     heat_dense,                          # (B,1,D,H,W)
            "quat":        quat_pred,                           # (B,4)
            "gripper":     gripper_log,                         # (B,1)

            # —— inference 用来拼成 (B,8) 动作
            "voxel_center": voxel_centers,                      # (B,3)

            # —— 供 loss.pose_to_heatmap() 使用的几项
            "xyz_min":     xyz_min_world,  # (B,3) 世界坐标系下的局部体素原点
            "grid_size":   torch.tensor([D, H, W], device=heat_dense.device),
            "voxel_size":  torch.tensor(self.voxel_size,
                                        device=heat_dense.device),   # (1,) NEW
        }

        if self.training:
            return output          # == dict 给 loss
        else:
            return self._dict2action(output)  # == (B,8)