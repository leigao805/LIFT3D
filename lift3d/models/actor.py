import abc
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn

from lift3d.helpers.graphics import PointCloud
from lift3d.models.mlp.batchnorm_mlp import BatchNormMLP
from lift3d.models.mlp.mlp import MLP


# lift3d/models/actor.py  (新增部分类)
# ---------------------------------------------------------------
from __future__ import annotations
import torch.nn.functional as F
from typing import Dict, Tuple

from lift3d.models.voxel_utils import tokens_to_sparse_voxel, to_me_tensor
from lift3d.models.sparse_unet import Sparse3DUNet
from lift3d.models.grasp_token_head import GraspOrientHead, GripperStateHead
from lift3d.models.lift3d_clip import Lift3dCLIP     # 仅用于类型提示

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


# ---------------------------------------------------------------
class TokenVoxelGraspActor(Actor):
    """
    Lift3D Stage‑2 策略网络 —— 使用:
    (1) Lift3dCLIP 输出的 patch‑token 特征+坐标
    (2) Sparse 3D U‑Net 生成粗抓取热图
    (3) Token 近邻特征回归抓取朝向 & 夹爪状态
    输出 dict:
        {
          'heatmap' : (B,1,D,H,W)  (dense),
          'quat'    : (B,4),       (单位四元数),
          'gripper' : (B,1)        (夹爪开概率 logits)
        }
    """

    def __init__(
        self,
        point_cloud_encoder: Lift3dCLIP,
        *,
        voxel_size: float = 0.01,
        pc_range: Tuple[float, float, float, float, float, float] = (
            -0.5, -0.5, -0.1, 0.5, 0.5, 0.4
        ),
        k_nearest: int = 4,
        sparse_unet_cfg: Dict | None = None,
        orient_head_cfg: Dict | None = None,
        gripper_head_cfg: Dict | None = None,
    ):
        super().__init__()
        # 1) 3D‑CLIP encoder (已含 LoRA / 冻结策略)
        self.point_cloud_encoder = point_cloud_encoder

        # 2) Voxel‑Net
        su_cfg = sparse_unet_cfg or {}
        self.sparse_unet = Sparse3DUNet(**su_cfg)

        # 3) Heads
        self.orient_head = GraspOrientHead(**(orient_head_cfg or {}))
        self.gripper_head = GripperStateHead(**(gripper_head_cfg or {}))

        # 其它超参
        self.voxel_size = voxel_size
        self.pc_range = list(pc_range)
        self.k_nearest = k_nearest

    # -----------------------------------------------------------
    def forward(
        self,
        images,
        point_clouds: torch.Tensor,         # (B, N, 3)
        robot_states,
        texts,
    ):
        """
        目前仅用 point_clouds; 保留其他参数以保持 Actor 接口一致
        """
        # ----- 1. Lift3D‑CLIP : 得到 CLS token + patch‑tokens + xyz -----
        cls_tok, patch_tok, patch_xyz = self.point_cloud_encoder(
            point_clouds, return_tokens=True, return_xyz=True
        )  # (B,768) (B,K,768) (B,K,3)

        # ----- 2. token -> sparse voxel -----
        coords, feats = tokens_to_sparse_voxel(
            patch_xyz, patch_tok,
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
        )
        sparse_tensor = to_me_tensor(coords, feats)        # Minkowski SparseTensor

        # ----- 3. Sparse 3D U‑Net -----
        heat_sparse = self.sparse_unet(sparse_tensor)      # SparseTensor (N',1)
        heat_dense  = heat_sparse.dense()                  # (B,1,D,H,W)

        # ----- 4. 选抓取点 (argmax 或 top‑k) -----
        B = heat_dense.size(0)
        flat = heat_dense.view(B, -1)
        _, idx = flat.max(dim=1)                           # (B,)
        # 把 voxel idx 反算到 xyz_center
        D, H, W = heat_dense.shape[-3:]
        z = (idx // (H * W)).int()
        y = ((idx % (H * W)) // W).int()
        x = (idx % W).int()
        voxel_centers = torch.stack([x, y, z], dim=1).float()  # (B,3)
        # scale back to meters
        voxel_centers = (voxel_centers + 0.5) * self.voxel_size + torch.tensor(
            self.pc_range[:3], device=voxel_centers.device
        )

        # ----- 5. 取最近 k 个 token 聚合 -----
        dist = torch.cdist(voxel_centers.unsqueeze(1), patch_xyz)  # (B,1,K)
        sel = dist.topk(self.k_nearest, largest=False).indices     # (B,1,k)
        sel = sel.expand(-1, patch_tok.size(-1), -1).transpose(1,2)  # (B,k,768) gather 用
        token_sel = torch.gather(patch_tok, dim=1, index=sel)      # (B,k,768)
        token_agg = token_sel.mean(dim=1)                          # (B,768)

        # ----- 6. Heads -----
        quat_pred   = self.orient_head(token_agg)   # (B,4)  已归一化
        gripper_log = self.gripper_head(token_agg)  # (B,1)  logits

        return {
            "heatmap": heat_dense,          # (B,1,D,H,W)
            "quat": quat_pred,              # (B,4)
            "gripper": gripper_log,         # (B,1)
            "cls_token": cls_tok,           # 可选：供多任务或对比损失
        }
