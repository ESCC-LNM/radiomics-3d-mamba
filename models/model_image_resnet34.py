"""
 3D ResNet 
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


DEFAULT_PRETRAIN_PATH = str((Path(__file__).resolve().parent / "resnet_34_23dataset.pth"))
DEFAULT_BASE_CHANNELS = 16
DEFAULT_STAGE_BLOCKS = (1, 1, 1, 1)
DEFAULT_BLOCK_DROPOUT = 0.10
DEFAULT_DROPOUT_RATE = 0.35


def _pick_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(_pick_group_count(num_channels, max_groups=max_groups), num_channels)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout_rate: float = DEFAULT_BLOCK_DROPOUT,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _make_group_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _make_group_norm(planes)
        self.downsample = downsample
        self.block_drop = nn.Dropout3d(p=float(dropout_rate)) if float(dropout_rate) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.block_drop(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet34_CT_Baseline(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        return_features_only: bool = False,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        block_dropout_rate: float = DEFAULT_BLOCK_DROPOUT,
    ):
        super().__init__()
        self.inplanes = int(base_channels)
        self.return_features_only = bool(return_features_only)
        self.block_dropout_rate = float(block_dropout_rate)

        self.conv1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = _make_group_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        stage_channels = [int(base_channels), int(base_channels) * 2, int(base_channels) * 4, int(base_channels) * 8]
        self.layer1 = self._make_layer(BasicBlock3D, stage_channels[0], blocks=DEFAULT_STAGE_BLOCKS[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, stage_channels[1], blocks=DEFAULT_STAGE_BLOCKS[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, stage_channels[2], blocks=DEFAULT_STAGE_BLOCKS[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, stage_channels[3], blocks=DEFAULT_STAGE_BLOCKS[3], stride=2)

        feat_dim = stage_channels[-1] * BasicBlock3D.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.head_drop = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feat_dim, num_classes)

        self._init_weights()

        log.info(
            f"[ResNet34_CT_Baseline] compact=True base_channels={base_channels} "
            f"blocks={DEFAULT_STAGE_BLOCKS} feat_dim={feat_dim} "
            f"block_dropout={self.block_dropout_rate} head_dropout={dropout_rate}"
        )

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                _make_group_norm(planes * block.expansion),
            )

        layers = [block(
            self.inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            dropout_rate=self.block_dropout_rate,
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes,
                planes,
                stride=1,
                downsample=None,
                dropout_rate=self.block_dropout_rate,
            ))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def forward(self, x_in: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x_in, dict):
            x = x_in.get("image", None)
            if x is None:
                raise KeyError("ResNet34_CT_Baseline expects dict with key 'image'.")
        else:
            x = x_in

        feat = self.forward_features(x)
        if self.return_features_only:
            return feat
        return self.fc(self.head_drop(feat))


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
        rep = weight.repeat(1, target_in_channels, 1, 1, 1) / float(target_in_channels)
        return rep

    if in_c > target_in_channels:
        return weight[:, :target_in_channels, ...]

    pad = target_in_channels - in_c
    extra = weight[:, :1, ...].repeat(1, pad, 1, 1, 1)
    return torch.cat([weight, extra], dim=1)


def load_pretrained_resnet34_3d(
    model: nn.Module,
    weights_path: str = DEFAULT_PRETRAIN_PATH,
    backbone_only: bool = True,
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
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        for prefix in ("model.", "backbone.", "resnet.", "encoder."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]

        if backbone_only and k2.startswith("fc."):
            continue
        if k2 not in model_state:
            continue

        if k2 == "conv1.weight" and v.shape != model_state[k2].shape:
            v = _adapt_conv1_weight_for_channels(v, target_in_channels=int(model_state[k2].shape[1]))

        if v.shape == model_state[k2].shape:
            new_state[k2] = v

    msg = model.load_state_dict(new_state, strict=strict)
    if print_info:
        log.info(f"[Pretrain] Loaded weights from: {weights_path}")
        log.info(f"[Pretrain] missing_keys: {msg.missing_keys}")
        log.info(f"[Pretrain] unexpected_keys: {msg.unexpected_keys}")


def create_resnet34_ct(
    num_classes: int = 2,
    in_channels: int = 2,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    backbone_only: bool = True,
    return_features_only: bool = False,
) -> ResNet34_CT_Baseline:
    model = ResNet34_CT_Baseline(
        in_channels=in_channels,
        num_classes=num_classes,
        return_features_only=return_features_only,
    )
    if pretrained:
        load_pretrained_resnet34_3d(
            model,
            weights_path=weights_path or DEFAULT_PRETRAIN_PATH,
            backbone_only=backbone_only,
            strict=False,
            print_info=True,
        )
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_resnet34_ct(
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
