"""
Radiomics-only comparator architecture.

This module defines the MLP used for the radiomics-only baseline, but it does
not provide a standalone manuscript-locked training CLI in this public release.
"""

import torch
import torch.nn as nn
from typing import Dict

class RadiomicsOnlyModel(nn.Module):
    """
    Input dict contract:
        {"rad_features": Tensor}
    Training output:
        {"fusion": logits, "img": None, "rad": None}
    Eval output:
        (logits, None, None)
    """
    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        x = batch["rad_features"]
        logits = self.mlp(x)
        if self.training:
            return {"fusion": logits, "img": None, "rad": None}
        return (logits, None, None)
