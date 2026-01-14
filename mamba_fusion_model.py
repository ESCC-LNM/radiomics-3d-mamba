"""
Mamba-Radiomics Fusion Model

Components:
- Mamba_CT: 3D CNN -> token sequence -> 2-layer Mamba -> pooled image embedding
- Radiomics MLP: tabular radiomics -> embedding -> radiomics baseline logits
- Fusion head: residual delta logits from concatenated (img_feat + rad_feat)
- Uncertainty gate: derived from radiomics confidence to modulate residual contribution

Return format:
- Training mode (model.training == True):
    dict with keys: {"fusion", "rad"} and optionally {"img"} if aux enabled
- Eval mode (model.training == False):
    tuple: (fusion_logits, u, gate) for interpretability analysis
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Union, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
# Library file: do NOT configure global logging here.
# Let the training script decide handlers/format.
if not log.handlers:
    log.addHandler(logging.NullHandler())

# =============================================================================
# Mamba dependency
# =============================================================================
try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "'mamba-ssm' is required. Install via: pip install mamba-ssm\n"
        "If you use CUDA, ensure you installed the correct build for your environment."
    ) from e


# =============================================================================
# 3D CNN feature extractor
# =============================================================================
class MambaCTFeatureExtractor(nn.Module):
    """
    Input : (B, 1, 32, 64, 64)
    Output: (B, embed_dim, 8, 16, 16)  (for default strides)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
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
            nn.Conv3d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_1x1 = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.proj_1x1(x)
        return x


# =============================================================================
# Mamba_CT backbone (2-layer Mamba)
# =============================================================================
class Mamba_CT(nn.Module):
    """
    Image encoder:
      3D CNN -> flatten spatial -> token sequence -> 2x Mamba -> average pool -> embedding
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        num_classes: int = 2,
        return_features_only: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.return_features_only = bool(return_features_only)

        self.feature_extractor = MambaCTFeatureExtractor(
            in_channels=in_channels,
            embed_dim=self.embed_dim,
        )

        # (B, C, D, H, W) -> (B, C, N) -> (B, N, C)
        self.flatten_spatial = nn.Flatten(start_dim=2)

        self.mamba_block1 = Mamba(
            d_model=self.embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.mamba_block2 = Mamba(
            d_model=self.embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # pool over sequence length N
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten_feat = nn.Flatten(start_dim=1)

        # optional classifier (unused when return_features_only=True)
        self.fc1 = nn.Linear(self.embed_dim, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> torch.Tensor:
        # (B, C, N) -> (B, N, C)
        return x.transpose(1, 2)

    def forward(self, x_in: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x_in, dict):
            x = x_in["image"]
        else:
            x = x_in

        feat3d = self.feature_extractor(x)                # (B, C, D', H', W')
        tokens = self.flatten_spatial(feat3d)             # (B, C, N)
        seq = self._to_seq(tokens)                        # (B, N, C)

        seq = self.mamba_block1(seq)
        seq = self.mamba_block2(seq)

        seq_t = seq.transpose(1, 2)                       # (B, C, N)
        pooled = self.avgpool(seq_t)                      # (B, C, 1)
        pooled = self.flatten_feat(pooled)                # (B, C)

        if self.return_features_only:
            return pooled

        x_fc = self.relu(self.fc1(pooled))
        logits = self.fc2(x_fc)
        return logits


# =============================================================================
# Fusion model (Radiomics baseline + gated residual delta logits)
# =============================================================================
class Mamba_Fusion_Model(nn.Module):
    """
    Training:
        returns dict {"fusion", "rad", optional "img"}
    Eval:
        returns (fusion_logits, u, gate) for interpretability
    """

    def __init__(
        self,
        mamba_embed_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        rad_input_features: int = 112,
        rad_hidden_features: int = 64,
        fusion_hidden_features: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_aux_cls: bool = True,
        aux_img_weight: float = 0.1,   # kept for config completeness
        aux_rad_weight: float = 0.05,  # kept for config completeness
        use_gated_fusion: bool = True,
        rad_scale: float = 0.3,
        gate_gamma: float = 2.0,
    ):
        super().__init__()

        self.use_aux_cls = bool(use_aux_cls)
        self.use_gated_fusion = bool(use_gated_fusion)
        self.rad_scale = float(rad_scale)
        self.gate_gamma = float(gate_gamma)
        self.num_classes = int(num_classes)

        # ---- Image encoder ----
        self.mamba_backbone = Mamba_CT(
            in_channels=1,
            embed_dim=mamba_embed_dim,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            num_classes=num_classes,
            return_features_only=True,
        )
        self.image_feature_dim = int(mamba_embed_dim)

        # ---- Radiomics branch ----
        self.radiomics_mlp = nn.Sequential(
            nn.Linear(rad_input_features, rad_hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.rad_feature_dim = int(rad_hidden_features)

        # Radiomics baseline head (also serves as auxiliary head)
        self.rad_head = nn.Linear(self.rad_feature_dim, num_classes)

        # Optional image auxiliary head
        self.img_head: Optional[nn.Module] = nn.Linear(self.image_feature_dim, num_classes) if self.use_aux_cls else None

        # ---- Fusion residual head: predicts delta logits ----
        fusion_in_dim = self.image_feature_dim + self.rad_feature_dim
        self.fusion_delta_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_features, num_classes),
        )

        # Do not force logging config here; just expose info for scripts if needed
        log.info(
            "[Mamba_Fusion_Model] ImgFeat=%d, RadFeat=%d, FusionIn=%d, Aux=%s, Gated=%s, RadScale=%.3f, GateGamma=%.3f",
            self.image_feature_dim,
            self.rad_feature_dim,
            fusion_in_dim,
            self.use_aux_cls,
            self.use_gated_fusion,
            self.rad_scale,
            self.gate_gamma,
        )

    # -------------------------------------------------------------------------
    # radiomics uncertainty -> (gate, u)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _compute_uncertainty_gate(self, logits_rad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sample-level gate from radiomics uncertainty:
          p = softmax(logits_rad)[:, 1]
          conf = |2p - 1|
          u = 1 - conf
          gate = u ^ gamma

        Returns:
          gate: (B, 1)
          u:    (B, 1)
        """
        p = logits_rad.softmax(dim=1)[:, 1:2]           # (B,1)
        conf = torch.abs(2.0 * p - 1.0)                 # (B,1)
        u = 1.0 - conf                                  # (B,1)
        gate = torch.pow(u, self.gate_gamma)            # (B,1)
        return gate, u

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not isinstance(batch_data, dict):
            raise TypeError("Expected a dict with keys: 'image', 'rad_features'.")
        if "image" not in batch_data or "rad_features" not in batch_data:
            raise KeyError("batch_data must contain both 'image' and 'rad_features'.")

        # Image embedding
        img_feat = self.mamba_backbone(batch_data)  # (B, C_img)

        # Radiomics embedding (scaled)
        rad_feat = self.radiomics_mlp(batch_data["rad_features"])  # (B, C_rad)
        rad_feat = rad_feat * self.rad_scale

        # Radiomics baseline logits
        logits_rad = self.rad_head(rad_feat)  # (B, num_classes)

        # Fusion residual delta logits
        concat_feat = torch.cat([img_feat, rad_feat], dim=1)
        delta_logits = self.fusion_delta_head(concat_feat)  # (B, num_classes)

        # Gate
        if self.use_gated_fusion:
            gate, u = self._compute_uncertainty_gate(logits_rad)  # (B,1)
        else:
            gate = torch.ones_like(logits_rad[:, :1])
            u = torch.zeros_like(logits_rad[:, :1])

        gate_expanded = gate.expand_as(logits_rad)  # (B, num_classes)

        # Final logits = radiomics baseline + gated residual
        fusion_logits = logits_rad + gate_expanded * delta_logits

        # Training: return dict (multi-head)
        if self.training:
            out: Dict[str, torch.Tensor] = {
                "fusion": fusion_logits,
                "rad": logits_rad,
            }
            if self.use_aux_cls and self.img_head is not None:
                out["img"] = self.img_head(img_feat)
            return out

        # Eval: return tuple for interpretability
        return fusion_logits, u, gate


# =============================================================================
# Minimal self-test (optional)
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mamba_Fusion_Model().to(device)

    batch = {
        "image": torch.randn(2, 1, 32, 64, 64, device=device),
        "rad_features": torch.randn(2, 112, device=device),
    }

    model.train()
    out_train = model(batch)
    print("Train keys:", sorted(list(out_train.keys())))

    model.eval()
    with torch.no_grad():
        out_eval = model(batch)
    print("Eval types:", type(out_eval), "len=", len(out_eval) if isinstance(out_eval, tuple) else "NA")
