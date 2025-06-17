import os
import pathlib
import sys

import torch

from lift3d.helpers.pytorch import log_params_to_file
from lift3d.models.lift3d.backbone.lift3d_clip import Lift3dCLIP
from lift3d.models.lift3d.model_utils.clip_loralib import (
    LoRALayer,
    apply_lora,
    merge_lora,
)
from lift3d.models.lift3d.model_utils.mv_utils import cfg_from_yaml_file


def set_trainable_params(model):
    substrings = ["cls_token", "cls_pos", "norm", "patch_embed", "patch_linear", "lora"]
    for n, p in model.named_parameters():
        p.requires_grad = True
        if all(sub not in n for sub in substrings):
            p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad = True
    return model


# ───── model_loader.py ──────────────────────────────────────────
def lift3d_clip_base(
    *,                                   ### NEW 允许 Hydra 传入关键字参数
    return_tokens: bool = False,         ### NEW
    return_xyz: bool = False,            ### NEW
):
    """
    Factory function that builds a Lift3d‑CLIP encoder ready for
    Stage‑2 imitation / RL.  Extra kwargs control the *default* behaviour
    of its forward() method (see lift3d_clip.py).

    Args
    ----
    return_tokens : if True, forward() 默认多返回 patch‑token 特征
    return_xyz    : 若 True，则再附带 patch 中心坐标
    """
    current_dir = pathlib.Path(__file__).parent
    yaml_path = os.path.join(current_dir, "model_config/ViT-B-32.yaml")
    config = cfg_from_yaml_file(yaml_path)
    # 1. 实例化基础 Lift3dCLIP
    model = Lift3dCLIP(config=config.model)
    # 2. Stage‑1 LoRA 预训练权重合并
    apply_lora(model)  # Stage 1: Apply LoRA for MAE pretraining

    ckpt_path = os.path.join(current_dir, "ckpt/lift3d_clip_base.pth")
    model.load_model_from_ckpt_mae(ckpt_path)
    merge_lora(model)

    # 3. Stage‑2 再插 LoRA，并冻结除指定层 + LoRA 以外的参数
    apply_lora(model)  # Stage 2: Apply LoRA for imitation learning
    set_trainable_params(model)

    # 4. 记录默认返回开关  -------------------------- ### NEW
    model.default_return_tokens = return_tokens
    model.default_return_xyz    = return_xyz
    #   （在 lift3d_clip.py 的 forward() 内部，若调用方
    #    没有显式传关键字参数，就会读取这两个属性）

    return model


def test_model_forward(model, logdir=None):
    """
    1. Test the model by performing a forward pass with random point cloud data.
    2. Log the trainable and frozen parameters to files if `logdir` is provided.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.float()
    pts = torch.randn([4, 1024, 3]).to(device)
    pts = pts.to(device)
    with torch.amp.autocast(device_type="cuda"):
        embedding_dim = model(pts).shape[1]
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        log_params_to_file(
            model, os.path.join(logdir, "trainable_params.txt"), requires_grad=True
        )
        log_params_to_file(
            model, os.path.join(logdir, "freeze_params.txt"), requires_grad=False
        )
    return model, embedding_dim


if __name__ == "__main__":
    from lift3d.helpers.common import Logger

    Logger.log_info(f"Test Lift3d-CLIP!!")
    model = lift3d_clip_base()
    model, embedding_dim = test_model_forward(model)
    assert embedding_dim == model.feature_dim
    Logger.log_info(f"feature_dim: {model.feature_dim}")
    Logger.print_seperator()
