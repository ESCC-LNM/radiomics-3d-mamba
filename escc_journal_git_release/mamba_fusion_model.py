"""
Manuscript-aligned multimodal fusion model.

Final fusion rule:
    fusion = (1 - alpha) * logits_rad + alpha * logits_img

where alpha in [0, 1] is a sample-wise scalar gate estimated from the
concatenated radiomics and image features.

Compatibility notes
-------------------
- Final training scripts use the keys: ``fusion``, ``img``, ``rad``.
- Optional preliminary CV utilities use the keys:
  ``fusion_logits``, ``image_logits``, ``radiomics_logits``.
- This module exposes both naming conventions to keep the repository coherent.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "mamba-ssm is required for the 3D Mamba fusion model. "
        "Install it in a CUDA-enabled environment before training or inference."
    ) from exc


def _require_cuda(x: torch.Tensor, where: str = "") -> None:
    if x.device.type != "cuda":
        raise RuntimeError(
            f"[Mamba CUDA-only] {where} received device={x.device}. "
            "Move both model and inputs to CUDA when using mamba-ssm kernels."
        )


class MambaCTFeatureExtractor(nn.Module):
    """3D convolutional image feature extractor.

    Input shape:
        (B, C, 32, 64, 64)
    Output shape:
        (B, 128, 8, 16, 16)
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Mamba_CT(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        num_classes: int = 2,
        return_features_only: bool = False,
        require_cuda: bool = True,
    ):
        super().__init__()
        self.return_features_only = bool(return_features_only)
        self.require_cuda = bool(require_cuda)

        self.feature_extractor = MambaCTFeatureExtractor(in_channels=in_channels, out_channels=embed_dim)
        self.flatten_spatial = nn.Flatten(start_dim=2)

        self.mamba_block1 = Mamba(d_model=embed_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.mamba_block2 = Mamba(d_model=embed_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten_feat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_in):
        x = x_in["image"] if isinstance(x_in, Mapping) else x_in
        if self.require_cuda:
            _require_cuda(x, where="Mamba_CT.forward(image)")

        feat3d = self.feature_extractor(x)
        tokens = self.flatten_spatial(feat3d)
        seq = tokens.transpose(1, 2).contiguous()

        if self.require_cuda:
            _require_cuda(seq, where="Mamba_CT.forward(seq)")

        seq_out = self.mamba_block1(seq)
        seq_out = self.mamba_block2(seq_out)
        seq_out_t = seq_out.transpose(1, 2).contiguous()
        pooled = self.flatten_feat(self.avgpool(seq_out_t))

        if self.return_features_only:
            return pooled

        logits = self.fc2(self.relu(self.fc1(pooled)))
        return logits


class Mamba_Fusion_Model(nn.Module):
    """Dual-branch radiomics–3D Mamba model with learnable gated logit fusion."""

    def __init__(
        self,
        mamba_embed_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        image_in_channels: int = 2,
        rad_input_features: Optional[int] = None,
        rad_hidden_features: Optional[int] = None,
        fusion_hidden_features: Optional[int] = None,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        use_aux_cls: bool = True,
        aux_img_weight: float = 0.25,
        aux_rad_weight: float = 0.0,
        rad_scale: float = 0.70,
        require_cuda: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Compatibility aliases used by the optional CV utilities.
        if rad_input_features is None:
            rad_input_features = int(kwargs.pop("radiomics_input_dim", 112))
        if rad_hidden_features is None:
            rad_hidden_features = int(kwargs.pop("radiomics_hidden_dim", 64))
        if fusion_hidden_features is None:
            fusion_hidden_features = int(kwargs.pop("fusion_hidden_dim", 64))
        if "use_auxiliary_image_head" in kwargs:
            use_aux_cls = bool(kwargs.pop("use_auxiliary_image_head"))
        if "radiomics_logit_scale" in kwargs:
            rad_scale = float(kwargs.pop("radiomics_logit_scale"))
        _ = aux_img_weight, aux_rad_weight  # retained only for call-site compatibility

        self.use_aux_cls = bool(use_aux_cls)
        self.rad_scale = float(rad_scale)
        self.num_classes = int(num_classes)
        self.require_cuda = bool(require_cuda)
        self.image_in_channels = int(image_in_channels)

        self.mamba_backbone = Mamba_CT(
            in_channels=self.image_in_channels,
            embed_dim=mamba_embed_dim,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            num_classes=num_classes,
            return_features_only=True,
            require_cuda=require_cuda,
        )
        self.image_feature_dim = int(mamba_embed_dim)

        self.radiomics_fc1 = nn.Linear(int(rad_input_features), int(rad_hidden_features))
        self.radiomics_relu = nn.ReLU(inplace=True)
        self.radiomics_drop = nn.Dropout(dropout_rate)
        self.rad_feature_dim = int(rad_hidden_features)

        self.rad_head = nn.Linear(self.rad_feature_dim, num_classes)
        self.img_head = nn.Linear(self.image_feature_dim, num_classes)

        fusion_in_dim = self.image_feature_dim + self.rad_feature_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_in_dim, int(fusion_hidden_features)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(int(fusion_hidden_features), 1),
        )

        log.info(
            "[Mamba_Fusion_Model] ImgInCh=%d, ImgFeat=%d, RadFeat=%d, FusionIn=%d, Aux=%s, ScalarGate=True, RadScale=%.3f",
            self.image_in_channels,
            self.image_feature_dim,
            self.rad_feature_dim,
            fusion_in_dim,
            self.use_aux_cls,
            self.rad_scale,
        )

    def forward(self, batch_data: Dict[str, torch.Tensor]):
        if not isinstance(batch_data, dict):
            raise ValueError("Expected dict with keys: 'image', 'rad_features'.")
        if "image" not in batch_data or "rad_features" not in batch_data:
            raise KeyError("batch_data must contain 'image' and 'rad_features'.")

        if self.require_cuda:
            _require_cuda(batch_data["image"], where="Mamba_Fusion_Model.forward(image)")
            _require_cuda(batch_data["rad_features"], where="Mamba_Fusion_Model.forward(rad_features)")

        img_feat = self.mamba_backbone({"image": batch_data["image"]})

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
            out = {
                "fusion": fusion_logits,
                "fusion_logits": fusion_logits,
                "rad": logits_rad,
                "radiomics_logits": logits_rad,
            }
            if self.use_aux_cls:
                out["img"] = logits_img
                out["image_logits"] = logits_img
            else:
                out["img"] = None
                out["image_logits"] = None
            return out

        return fusion_logits, logits_img, logits_rad, u, gate


# Compatibility alias used by the optional CV utilities.
MambaFusionModel = Mamba_Fusion_Model


if __name__ == "__main__":
    print("This module defines the manuscript-aligned radiomics–3D Mamba fusion model.")
