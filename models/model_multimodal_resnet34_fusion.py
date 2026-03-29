"""
ResNet Fusion Model (3D CT + Radiomics)
"""

from __future__ import annotations

import os
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from models.model_image_resnet34 import generate_model

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

DEFAULT_PRETRAIN_PATH = str((Path(__file__).resolve().parent / "resnet_34_23dataset.pth"))


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise TypeError("Unsupported checkpoint format: expected dict-like checkpoint/state_dict.")


def _adapt_conv1_weight_for_channels(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    in_c = int(weight.shape[1])
    if in_c == target_in_channels:
        return weight

    if in_c == 1 and target_in_channels > 1:
        return weight.repeat(1, target_in_channels, 1, 1, 1) / float(target_in_channels)

    if in_c > target_in_channels:
        return weight[:, :target_in_channels, ...]

    pad = target_in_channels - in_c
    extra = weight[:, :1, ...].repeat(1, pad, 1, 1, 1)
    return torch.cat([weight, extra], dim=1)


class ResNet_Fusion_Model(nn.Module):
    """
    Training: returns dict {"fusion", "rad", optional "img"}
    Eval: returns (fusion_logits, u, gate)
    """

    def __init__(
        self,
        image_in_channels: int = 2,
        rad_input_features: int = 112,
        rad_hidden_features: int = 64,
        fusion_hidden_features: int = 64,
        embed_dim: int = 128,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_pretrained: bool = False,
        pretrained_path: str = DEFAULT_PRETRAIN_PATH,
        use_aux_cls: bool = True,
        rad_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs

        self.use_aux_cls = bool(use_aux_cls)
        self.rad_scale = float(rad_scale)
        self.num_classes = int(num_classes)
        self.image_in_channels = int(image_in_channels)

        self.image_backbone, in_f = self._build_image_backbone(
            image_in_channels=self.image_in_channels,
            num_classes=self.num_classes,
        )

        if bool(use_pretrained):
            self._load_pretrained_backbone(
                self.image_backbone,
                weights_path=pretrained_path,
                target_in_channels=self.image_in_channels,
            )

        self.img_proj = nn.Sequential(
            nn.Linear(in_f, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.image_feature_dim = int(embed_dim)

        self.radiomics_fc1 = nn.Linear(rad_input_features, rad_hidden_features)
        self.radiomics_relu = nn.ReLU(inplace=True)
        self.radiomics_drop = nn.Dropout(dropout_rate)
        self.rad_feature_dim = int(rad_hidden_features)

        self.rad_head = nn.Linear(self.rad_feature_dim, num_classes)
        self.img_head = nn.Linear(self.image_feature_dim, num_classes)

        fusion_in_dim = self.image_feature_dim + self.rad_feature_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_features, 1),
        )

        log.info(
            f"[ResNet_Fusion_Model] ImgInCh={self.image_in_channels}, ImgFeat={self.image_feature_dim}, "
            f"RadFeat={self.rad_feature_dim}, FusionIn={fusion_in_dim}, Aux={self.use_aux_cls}, "
            f"LearnableConvexGate=True, ScalarGate=True, RadMLP=1Hidden, "
            f"RadScale={self.rad_scale}, Pretrained={bool(use_pretrained)}"
        )

    def forward(self, batch_data: Dict[str, torch.Tensor]):
        if not isinstance(batch_data, dict):
            raise ValueError("Expected dict with keys: 'image', 'rad_features'.")
        if "image" not in batch_data or "rad_features" not in batch_data:
            raise KeyError("batch_data must contain 'image' and 'rad_features'.")

        img_raw = self.image_backbone(batch_data["image"])
        if img_raw.ndim > 2:
            img_raw = img_raw.view(img_raw.size(0), -1)
        img_feat = self.img_proj(img_raw)

        rad_feat = self.radiomics_fc1(batch_data["rad_features"])
        rad_feat = self.radiomics_relu(rad_feat)
        rad_feat = self.radiomics_drop(rad_feat)
        rad_feat = rad_feat * self.rad_scale

        logits_rad = self.rad_head(rad_feat)
        logits_img = self.img_head(img_feat)

        concat_feat = torch.cat([img_feat, rad_feat], dim=1)
        alpha = torch.sigmoid(self.fusion_gate(concat_feat))
        alpha_expanded = alpha.expand_as(logits_rad)
        fusion_logits = (1.0 - alpha_expanded) * logits_rad + alpha_expanded * logits_img

        gate = alpha
        u = 1.0 - alpha

        if self.training:
            out = {"fusion": fusion_logits, "rad": logits_rad}
            if self.use_aux_cls:
                out["img"] = logits_img
            return out

        return fusion_logits, u, gate

    @staticmethod
    def _build_image_backbone(image_in_channels: int, num_classes: int) -> Tuple[nn.Module, int]:
        sig = inspect.signature(generate_model)
        kwargs: Dict[str, Any] = {}

        if "model_depth" in sig.parameters:
            kwargs["model_depth"] = 34
            if "n_input_channels" in sig.parameters:
                kwargs["n_input_channels"] = int(image_in_channels)
            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = int(num_classes)
            if "sample_size" in sig.parameters:
                kwargs["sample_size"] = 64
            if "sample_duration" in sig.parameters:
                kwargs["sample_duration"] = 32
            backbone = generate_model(**kwargs)
        else:
            backbone = generate_model(34)

        return ResNet_Fusion_Model._strip_classifier(backbone)

    @staticmethod
    def _strip_classifier(backbone: nn.Module) -> Tuple[nn.Module, int]:
        if hasattr(backbone, "fc"):
            in_f = int(backbone.fc.in_features)
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "last_linear"):
            in_f = int(backbone.last_linear.in_features)
            backbone.last_linear = nn.Identity()
        else:
            in_f = int(getattr(backbone, "feature_dim", 512))
        return backbone, in_f

    @staticmethod
    def _load_pretrained_backbone(backbone: nn.Module, weights_path: str, target_in_channels: int) -> None:
        if not weights_path or not os.path.isfile(weights_path):
            log.warning(f"[Pretrain] Weights not found, skip loading: {weights_path}")
            return

        ckpt = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(ckpt)

        model_state = backbone.state_dict()
        new_state: Dict[str, torch.Tensor] = {}

        for k, v in state.items():
            k2 = k
            if k2.startswith("module."):
                k2 = k2[len("module.") :]
            for prefix in ("model.", "backbone.", "resnet.", "encoder."):
                if k2.startswith(prefix):
                    k2 = k2[len(prefix) :]

            if k2 not in model_state:
                continue

            if k2 == "conv1.weight" and v.shape != model_state[k2].shape:
                v = _adapt_conv1_weight_for_channels(v, target_in_channels=target_in_channels)

            if v.shape == model_state[k2].shape:
                new_state[k2] = v

        msg = backbone.load_state_dict(new_state, strict=False)
        log.info(f"[Pretrain] Loaded backbone weights from: {weights_path}")
        log.info(f"[Pretrain] missing_keys: {msg.missing_keys}")
        log.info(f"[Pretrain] unexpected_keys: {msg.unexpected_keys}")
