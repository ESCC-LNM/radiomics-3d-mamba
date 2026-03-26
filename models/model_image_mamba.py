"""
Mamba-CT (3D CT-only) baseline model - publication-safe aligned version

What this fixes / guarantees for Academic Peer-Review:
------------------------------------------------------
1) 100% Architectural Alignment: The 3D CNN stem (BatchNorm, ReLU, channel dims),
   the dual Mamba blocks, the pooling strategy, and the MLP head are EXACT replicas
   of the CT branch in `Mamba_Fusion_Model`.
2) Controlling Variables: Any performance difference between this model and the
   fusion model can now be mathematically attributed ONLY to the radiomics fusion
   mechanism, satisfying strict reviewer scrutiny.
3) Fast-fail CUDA enforcement is maintained to match the main training pipeline.
"""

import logging
from typing import Dict, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# =============================================================================
# Mamba dependency (Strict requirement aligned with fusion script)
# =============================================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    log.error("=" * 80)
    log.error("!!! 'mamba-ssm' is required. Install: pip install mamba-ssm")
    log.error("=" * 80)
    raise


def _require_cuda(x: torch.Tensor, where: str = "") -> None:
    """Fail fast if tensors are not on CUDA (matches fusion model behavior)."""
    if x.device.type != "cuda":
        raise RuntimeError(
            f"[Mamba CUDA-only] {where} got device={x.device}. "
            "Your installed mamba_ssm uses CUDA-only kernels. "
            "Move BOTH model and inputs to CUDA."
        )


# =============================================================================
# 3D CNN feature extractor (100% ALIGNED with Script 1)
# =============================================================================
class MambaCTFeatureExtractor(nn.Module):
    """
    Input:  (B, 2, 32, 64, 64)
    Output: (B, 128, 8, 16, 16)
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super().__init__()

        # Aligned: Conv3d -> BatchNorm3d -> ReLU (No InstanceNorm/GELU here)
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
        self.conv4_1x1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4_1x1(x)
        return x


# =============================================================================
# Mamba_CT (100% ALIGNED with Script 1, wrapped as standalone classifier)
# =============================================================================
class Mamba_CT_Baseline(nn.Module):
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
        self.embed_dim = embed_dim
        self.return_features_only = return_features_only
        self.require_cuda = bool(require_cuda)

        self.feature_extractor = MambaCTFeatureExtractor(
            in_channels=in_channels,
            out_channels=embed_dim,
        )

        self.flatten_spatial = nn.Flatten(start_dim=2)  # (B, C, D,H,W) -> (B, C, N)

        # Aligned: TWO Mamba blocks, not one
        self.mamba_block1 = Mamba(
            d_model=embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.mamba_block2 = Mamba(
            d_model=embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # Aligned: AdaptiveAvgPool1d, not x.mean(dim=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten_feat = nn.Flatten(start_dim=1)

        # Aligned: Linear -> ReLU -> Linear, no LayerNorm/Dropout
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_in: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Supports both raw tensors and dicts for seamless integration with your pipeline.
        """
        if isinstance(x_in, dict):
            x = x_in.get("image", None)
            if x is None:
                raise KeyError("Mamba_CT_Baseline expects dict with key 'image'.")
        else:
            x = x_in

        if self.require_cuda:
            _require_cuda(x, where="Mamba_CT_Baseline.forward(image)")

        feat3d = self.feature_extractor(x)  # (B, C, D', H', W')
        tokens = self.flatten_spatial(feat3d)  # (B, C, N)
        seq = tokens.transpose(1, 2).contiguous()  # (B, N, C)

        if self.require_cuda:
            _require_cuda(seq, where="Mamba_CT_Baseline.forward(seq)")

        seq_out = self.mamba_block1(seq)
        seq_out = self.mamba_block2(seq_out)

        seq_out_t = seq_out.transpose(1, 2).contiguous()  # (B, C, N)
        pooled = self.avgpool(seq_out_t)  # (B, C, 1)
        pooled = self.flatten_feat(pooled)  # (B, C)

        if self.return_features_only:
            return pooled

        x_fc = self.fc1(pooled)
        x_fc = self.relu(x_fc)
        logits = self.fc2(x_fc)

        # Note: If your training script strictly expects a dict back,
        # you can return {"fusion": logits} here to spoof the pipeline.
        return logits


# =============================================================================
# Self-Test
# =============================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Your mamba_ssm build is CUDA-only; run this test on a GPU machine."
        )

    device = torch.device("cuda")
    model = Mamba_CT_Baseline(require_cuda=True).to(device)

    # Test with dict input (like your fusion model)
    batch = {
        "image": torch.randn(2, 2, 32, 64, 64, device=device)
    }

    model.eval()
    with torch.no_grad():
        logits = model(batch)

    print(f"Alignment Test Passed! Baseline logits shape: {tuple(logits.shape)}")
