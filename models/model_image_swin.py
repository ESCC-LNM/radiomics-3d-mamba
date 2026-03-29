"""
Swin image baseline 
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


DEFAULT_PRETRAIN_PATH = str((Path(__file__).resolve().parent / "pytorch_model.bin"))
DEFAULT_ROI_SIZE = (32, 64, 64)
DEFAULT_FEATURE_SIZE = 12
DEFAULT_DEPTHS = (1, 1, 1, 1)
DEFAULT_NUM_HEADS = (3, 3, 6, 6)
DEFAULT_PROJ_DIM = 96
DEFAULT_HEAD_DROPOUT = 0.30


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise TypeError("Unsupported checkpoint format: expected dict-like checkpoint/state_dict.")


def _clean_key(k: str) -> str:
    k2 = k
    if k2.startswith("module."):
        k2 = k2[len("module."):]
    for prefix in ("model.", "backbone.", "encoder.", "swinViT."):
        if k2.startswith(prefix):
            k2 = k2[len(prefix):]
    return k2


def _adapt_conv_weight_if_needed(v: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
    if v.ndim != 5 or len(target_shape) != 5:
        return None
    if tuple(v.shape) == tuple(target_shape):
        return v

    in_c_src = int(v.shape[1])
    in_c_dst = int(target_shape[1])
    if in_c_src == in_c_dst:
        return v if tuple(v.shape) == tuple(target_shape) else None

    if in_c_src == 1 and in_c_dst > 1 and v.shape[0] == target_shape[0] and tuple(v.shape[2:]) == tuple(target_shape[2:]):
        vv = v.repeat(1, in_c_dst, 1, 1, 1) / float(in_c_dst)
        return vv
    return None


def load_pretrained_swinunetr_3d(
    model: nn.Module,
    weights_path: str = DEFAULT_PRETRAIN_PATH,
    strict: bool = False,
    print_info: bool = True,
) -> None:
    if not weights_path or not os.path.isfile(weights_path):
        if print_info:
            log.warning(f"[Pretrain] Weights not found, skip loading: {weights_path}")
        return

    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    model_state = model.state_dict()
    new_state = {}
    for k, v in state.items():
        k2 = _clean_key(k)
        if k2 not in model_state:
            continue

        tgt = model_state[k2]
        if tuple(v.shape) == tuple(tgt.shape):
            new_state[k2] = v
            continue

        v2 = _adapt_conv_weight_if_needed(v, tgt.shape)
        if v2 is not None and tuple(v2.shape) == tuple(tgt.shape):
            new_state[k2] = v2

    msg = model.load_state_dict(new_state, strict=strict)
    if print_info:
        log.info(f"[Pretrain] Loaded weights from: {weights_path}")
        log.info(f"[Pretrain] missing_keys: {msg.missing_keys}")
        log.info(f"[Pretrain] unexpected_keys: {msg.unexpected_keys}")


def create_swin_encoder_from_swinunetr_ckpt(
    weights_path: Optional[str],
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    in_channels: int = 2,
    feature_size: int = DEFAULT_FEATURE_SIZE,
    use_checkpoint: bool = True,
    depths: Tuple[int, int, int, int] = DEFAULT_DEPTHS,
    num_heads: Tuple[int, int, int, int] = DEFAULT_NUM_HEADS,
) -> nn.Module:
    sig = inspect.signature(SwinUNETR)
    ctor_kwargs: Dict[str, Any] = {
        "img_size": roi_size,
        "in_channels": in_channels,
        "out_channels": 2,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
    }
    if "depths" in sig.parameters:
        ctor_kwargs["depths"] = depths
    if "num_heads" in sig.parameters:
        ctor_kwargs["num_heads"] = num_heads

    temp = SwinUNETR(**ctor_kwargs)

    if weights_path:
        load_pretrained_swinunetr_3d(
            temp,
            weights_path=weights_path,
            strict=False,
            print_info=True,
        )

    encoder = temp.swinViT
    del temp
    return encoder


class SwinCT_Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 2,
        roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
        feature_size: int = DEFAULT_FEATURE_SIZE,
        use_checkpoint: bool = True,
        pretrained: bool = False,
        weights_path: str = DEFAULT_PRETRAIN_PATH,
        return_features_only: bool = False,
        proj_dim: int = DEFAULT_PROJ_DIM,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        depths: Tuple[int, int, int, int] = DEFAULT_DEPTHS,
        num_heads: Tuple[int, int, int, int] = DEFAULT_NUM_HEADS,
    ):
        super().__init__()
        self.return_features_only = bool(return_features_only)

        backbone_weights = weights_path if pretrained else None
        self.backbone = create_swin_encoder_from_swinunetr_ckpt(
            weights_path=backbone_weights,
            roi_size=roi_size,
            in_channels=in_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            depths=depths,
            num_heads=num_heads,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *roi_size)
            feats = self.backbone(dummy)
            last = feats[-1] if isinstance(feats, (list, tuple)) else feats
            feat_dim = int(last.shape[1])

        compact_dim = min(int(proj_dim), feat_dim)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, compact_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.fc = nn.Linear(compact_dim, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

        log.info(
            f"[SwinCT] compact=True roi={roi_size} in_channels={in_channels} "
            f"feature_size={feature_size} depths={depths} num_heads={num_heads} "
            f"feat_dim={feat_dim} compact_dim={compact_dim} pretrained={pretrained}"
        )

    def forward(self, x_in: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x_in, dict):
            x = x_in.get("image", None)
            if x is None:
                raise KeyError("SwinCT_Baseline expects dict with key 'image'.")
        else:
            x = x_in

        feats = self.backbone(x)
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats
        pooled = self.avgpool(x)
        pooled = self.flatten(pooled)
        pooled = self.feature_proj(pooled)

        if self.return_features_only:
            return pooled

        return self.fc(pooled)


def create_swin_ct(
    num_classes: int = 2,
    in_channels: int = 2,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    roi_size: Tuple[int, int, int] = DEFAULT_ROI_SIZE,
    feature_size: int = DEFAULT_FEATURE_SIZE,
    use_checkpoint: bool = True,
    return_features_only: bool = False,
) -> SwinCT_Baseline:
    return SwinCT_Baseline(
        num_classes=num_classes,
        in_channels=in_channels,
        roi_size=roi_size,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        pretrained=pretrained,
        weights_path=weights_path or DEFAULT_PRETRAIN_PATH,
        return_features_only=return_features_only,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_swin_ct(
        num_classes=2,
        in_channels=2,
        pretrained=False,
        return_features_only=False,
    ).to(device)
    model.eval()

    x = torch.randn(2, 2, 32, 64, 64, device=device)
    batch = {"image": x}

    with torch.no_grad():
        logits = model(batch)

    print("OK logits:", tuple(logits.shape))
