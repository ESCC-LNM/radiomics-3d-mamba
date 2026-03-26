"""
Swin Fusion Model (3D CT + Radiomics)

Fair-comparison design against Radiomics_3D_Mamba:
1. Keep the same fusion head/protocol (learnable convex gate).
2. Keep the same trainer I/O contract.
3. Only change image encoder: Swin instead of Mamba encoder.

Compatibility with trainer:
- train(): returns dict {"fusion", "rad", optional "img"}
- eval(): returns tuple (fusion_logits, u, gate), where gate=alpha, u=1-alpha
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.model_image_swin import create_swin_ct

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

DEFAULT_PRETRAIN_PATH = str((Path(__file__).resolve().parent / "pytorch_model.bin"))


class Swin_Fusion_Model(nn.Module):
    """
    Training: returns dict {"fusion", "rad", optional "img"}
    Eval: returns (fusion_logits, u, gate)
    """

    def __init__(
        self,
        image_in_channels: int = 2,
        roi_size: Tuple[int, int, int] = (32, 64, 64),
        use_pretrained: bool = False,
        weights_path: str = DEFAULT_PRETRAIN_PATH,
        rad_input_features: int = 112,
        rad_hidden_features: int = 64,
        fusion_hidden_features: int = 64,
        embed_dim: int = 128,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_aux_cls: bool = True,
        rad_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs

        self.use_aux_cls = bool(use_aux_cls)
        self.rad_scale = float(rad_scale)
        self.num_classes = int(num_classes)
        self.image_in_channels = int(image_in_channels)

        self.image_backbone = create_swin_ct(
            num_classes=self.num_classes,
            in_channels=self.image_in_channels,
            pretrained=bool(use_pretrained),
            weights_path=str(weights_path),
            roi_size=tuple(int(x) for x in roi_size),
            return_features_only=True,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.image_in_channels, *tuple(int(x) for x in roi_size))
            feat = self.image_backbone({"image": dummy})
            if feat.ndim > 2:
                feat = feat.view(feat.size(0), -1)
            img_in_dim = int(feat.shape[1])

        if img_in_dim == int(embed_dim):
            self.img_proj = nn.Identity()
            self.image_feature_dim = int(embed_dim)
        else:
            self.img_proj = nn.Sequential(
                nn.Linear(img_in_dim, int(embed_dim)),
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
            f"[Swin_Fusion_Model] ImgInCh={self.image_in_channels}, ImgFeat={self.image_feature_dim}, "
            f"RadFeat={self.rad_feature_dim}, FusionIn={fusion_in_dim}, Aux={self.use_aux_cls}, "
            f"LearnableConvexGate=True, ScalarGate=True, RadMLP=1Hidden, "
            f"RadScale={self.rad_scale}, Pretrained={bool(use_pretrained)}"
        )

    def forward(self, batch_data: Dict[str, torch.Tensor]):
        if not isinstance(batch_data, dict):
            raise ValueError("Expected dict with keys: 'image', 'rad_features'.")
        if "image" not in batch_data or "rad_features" not in batch_data:
            raise KeyError("batch_data must contain 'image' and 'rad_features'.")

        img_feat_raw = self.image_backbone({"image": batch_data["image"]})
        if img_feat_raw.ndim > 2:
            img_feat_raw = img_feat_raw.view(img_feat_raw.size(0), -1)
        img_feat = self.img_proj(img_feat_raw)

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
