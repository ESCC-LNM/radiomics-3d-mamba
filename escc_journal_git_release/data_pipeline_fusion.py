"""
Development-stage data pipeline for optional preliminary 5-fold tuning.

This module is not the final manuscript-evaluation pipeline. It is limited to
patient-level outer cross-validation within the development cohort and is meant
for preliminary hyperparameter / epoch selection only.

The image branch is aligned with the final manuscript implementation:
- image input is dual-channel [CT, binary mask]
- radiomics imputation and scaling are fit on the fold-train subset only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from experiment_utils import (
    get_optional,
    get_required,
    normalize_sample_id,
    require_columns,
    resolve_image_path,
    seed_worker,
)


def _read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
    require_columns(df, ["sample_id", "patient_id", "label", "outer_fold", "fold_role", "cohort_role"], f"Manifest '{path.name}'")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).map(normalize_sample_id)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["outer_fold"] = df["outer_fold"].astype(int)
    return df


def _read_selected_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "ID"})
    df["ID"] = df["ID"].astype(str).map(normalize_sample_id)
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].tolist()[:10]
        raise ValueError(f"Duplicate radiomics sample identifiers detected in {path.name}. Examples: {dup}")
    feature_cols = [c for c in df.columns if c != "ID"]
    if not feature_cols:
        raise ValueError(f"No selected radiomics feature columns found in {path.name}")
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df.set_index("ID")


def _resolve_mask_path(mask_dir: Path, sample_id: str, allowed_suffixes: List[str]) -> Path:
    stems = [
        sample_id,
        f"{sample_id}_mask",
        f"{sample_id}_seg",
        f"{sample_id}_label",
        f"{sample_id}-mask",
        f"{sample_id}-seg",
        f"{sample_id}-label",
    ]
    for stem in stems:
        for suffix in allowed_suffixes:
            candidate = mask_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve a mask file for sample '{sample_id}' in '{mask_dir}'.")


def _build_base_transform(cfg: Mapping[str, Any], training: bool = False) -> Compose:
    roi_size = tuple(int(v) for v in get_required(cfg, "preprocessing.image.roi_size"))
    intensity_window = get_required(cfg, "preprocessing.image.intensity_window")
    a_min = float(intensity_window[0])
    a_max = float(intensity_window[1])
    transforms: List[Any] = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys="image", a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
        Resized(keys="mask", spatial_size=roi_size, mode="nearest"),
    ]
    if training:
        aug_cfg = get_optional(cfg, "preprocessing.image.augmentation", default={}) or {}
        if bool(aug_cfg.get("enable_random_flip", True)):
            transforms.append(RandFlipd(keys=["image", "mask"], prob=float(aug_cfg.get("flip_probability", 0.5)), spatial_axis=[0, 1, 2]))
        if bool(aug_cfg.get("enable_random_rotate90", True)):
            transforms.append(RandRotate90d(keys=["image", "mask"], prob=float(aug_cfg.get("rotate90_probability", 0.5)), max_k=3, spatial_axes=(0, 1)))
        if bool(aug_cfg.get("enable_random_affine", True)):
            transforms.append(
                RandAffined(
                    keys=["image", "mask"],
                    prob=float(aug_cfg.get("affine_probability", 0.3)),
                    rotate_range=tuple(float(v) for v in aug_cfg.get("rotate_range", [0.2, 0.2, 0.2])),
                    scale_range=tuple(float(v) for v in aug_cfg.get("scale_range", [0.1, 0.1, 0.1])),
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                )
            )
    transforms.append(ToTensord(keys=["image", "mask"]))
    return Compose(transforms)


class _FusionDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, Any]], *, base_transform: Compose, cache_rate: float):
        self.rows = rows
        self.base_transform = base_transform
        bounded_cache_rate = max(0.0, min(1.0, float(cache_rate)))
        cache_count = int(len(rows) * bounded_cache_rate)
        if bounded_cache_rate > 0.0 and cache_count == 0 and rows:
            cache_count = 1
        self.cached_rows = [self._prepare_row(row) for row in rows[:cache_count]]

    def _prepare_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        transformed = self.base_transform({"image": row["image"], "mask": row["mask"]})
        image = transformed["image"]
        mask = (transformed["mask"] > 0).to(dtype=image.dtype)
        image = torch.cat([image, mask], dim=0)
        return {
            "image": image,
            "rad_features": torch.tensor(row["rad_features"], dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
        }

    @staticmethod
    def _clone_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": row["image"].clone(),
            "rad_features": row["rad_features"].clone(),
            "label": row["label"].clone(),
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_row = self.cached_rows[index] if index < len(self.cached_rows) else self._prepare_row(self.rows[index])
        record = self._clone_row(base_row)
        return record


def _build_rows(meta_rows: pd.DataFrame, rad_array, image_dir: Path, mask_dir: Path, suffixes: List[str]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(meta_rows.reset_index(drop=True).iterrows()):
        sample_id = str(row["sample_id"])
        output.append(
            {
                "image": str(resolve_image_path(image_dir, sample_id, suffixes)),
                "mask": str(_resolve_mask_path(mask_dir, sample_id, suffixes)),
                "rad_features": rad_array[i],
                "label": int(row["label"]),
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
            }
        )
    return output


def build_cv_dataloaders(cfg: Mapping[str, Any], outer_fold: int) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    preprocessed_root = Path(get_required(cfg, "paths.preprocessed_root"))
    manifest = _read_manifest(preprocessed_root / "cv_split_manifest.csv")
    fold_rows = manifest.loc[(manifest["cohort_role"] == "development") & (manifest["outer_fold"] == int(outer_fold))].copy()
    train_rows = fold_rows.loc[fold_rows["fold_role"] == "train"].copy()
    val_rows = fold_rows.loc[fold_rows["fold_role"] == "val"].copy()
    if train_rows.empty or val_rows.empty:
        raise ValueError(f"Outer fold {outer_fold} does not contain both train and val rows.")

    selected_root = Path(get_required(cfg, "paths.selected_features_root")) / "folds" / f"fold_{int(outer_fold):02d}"
    train_table = _read_selected_table(selected_root / "train_selected.csv")
    val_table = _read_selected_table(selected_root / "val_selected.csv")

    train_sample_ids = train_rows["sample_id"].astype(str).tolist()
    val_sample_ids = val_rows["sample_id"].astype(str).tolist()
    missing_train = [sid for sid in train_sample_ids if sid not in train_table.index]
    missing_val = [sid for sid in val_sample_ids if sid not in val_table.index]
    if missing_train:
        raise RuntimeError(f"Fold {outer_fold} train radiomics table is missing samples. Examples: {missing_train[:10]}")
    if missing_val:
        raise RuntimeError(f"Fold {outer_fold} val radiomics table is missing samples. Examples: {missing_val[:10]}")

    train_table = train_table.loc[train_sample_ids].copy()
    val_table = val_table.loc[val_sample_ids].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_rad = scaler.fit_transform(imputer.fit_transform(train_table))
    val_rad = scaler.transform(imputer.transform(val_table))

    image_dir = Path(get_required(cfg, "paths.images.development_dir"))
    mask_dir = Path(get_required(cfg, "paths.masks.development_dir"))
    suffixes = [str(v) for v in get_required(cfg, "dataset.allowed_image_suffixes")]
    train_records = _build_rows(train_rows, train_rad, image_dir, mask_dir, suffixes)
    val_records = _build_rows(val_rows, val_rad, image_dir, mask_dir, suffixes)

    use_cache = bool(get_optional(cfg, "runtime.use_cache_dataset", default=True))
    cache_rate = float(get_optional(cfg, "runtime.cache_rate", default=1.0)) if use_cache else 0.0
    batch_size = int(get_required(cfg, "runtime.batch_size"))
    eval_batch_size = int(get_optional(cfg, "runtime.eval_batch_size", default=batch_size))
    num_workers = int(get_optional(cfg, "runtime.num_workers", default=0))
    pin_memory = bool(get_optional(cfg, "runtime.pin_memory", default=True))
    train_transform = _build_base_transform(cfg, training=True)
    val_transform = _build_base_transform(cfg, training=False)

    train_dataset = _FusionDataset(train_records, base_transform=train_transform, cache_rate=cache_rate)
    val_dataset = _FusionDataset(val_records, base_transform=val_transform, cache_rate=cache_rate)

    generator = torch.Generator()
    generator.manual_seed(int(get_required(cfg, "cross_validation.seed")) + int(outer_fold))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=generator)

    metadata = {
        "outer_fold": int(outer_fold),
        "train_n": int(len(train_records)),
        "val_n": int(len(val_records)),
        "radiomics_input_dim": int(train_table.shape[1]),
        "train_patient_n": int(train_rows["patient_id"].nunique()),
        "val_patient_n": int(val_rows["patient_id"].nunique()),
    }
    return {"train": train_loader, "val": val_loader}, metadata
