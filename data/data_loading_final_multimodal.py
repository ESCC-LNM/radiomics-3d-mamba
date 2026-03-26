"""
Final manuscript-aligned data pipeline for multimodal fusion.

This module builds the pooled development training loader plus held-out internal
and external evaluation loaders. It is configuration-driven and does not ship
with site-specific filesystem defaults.
"""

from __future__ import annotations

import argparse
import logging
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

from utils.analysis_utils import (
    configure_logging,
    get_optional,
    get_required,
    load_json,
    normalize_patient_id,
    normalize_sample_id,
    require_columns,
    resolve_image_path,
    seed_worker,
)

log = logging.getLogger("fusion_journal_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final manuscript-aligned multimodal dataloaders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    return parser.parse_args()


def _read_internal_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))

    path = Path(get_required(cfg, "paths.internal_metadata_csv"))
    df = pd.read_csv(path, dtype={sample_id_col: str, patient_id_col: str})
    require_columns(df, [sample_id_col, patient_id_col, label_col, group_col], f"Metadata file '{path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    df[patient_id_col] = df[patient_id_col].astype(str).map(normalize_patient_id)
    df[label_col] = df[label_col].astype(int)
    df[group_col] = df[group_col].astype(str).str.strip()
    return df


def _read_external_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    external_patient_id_col = get_optional(cfg, "columns.external_patient_id", default=None)
    path = Path(get_required(cfg, "paths.external_metadata_csv"))
    dtypes = {sample_id_col: str}
    if external_patient_id_col is not None:
        dtypes[str(external_patient_id_col)] = str
    df = pd.read_csv(path, dtype=dtypes)
    required_cols = [sample_id_col, label_col]
    if external_patient_id_col is not None:
        required_cols.append(str(external_patient_id_col))
    require_columns(df, required_cols, f"Metadata file '{path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    df[label_col] = df[label_col].astype(int)
    if external_patient_id_col is not None:
        df[str(external_patient_id_col)] = df[str(external_patient_id_col)].astype(str).map(normalize_patient_id)
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


def _resolve_final_selected_paths(cfg: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    final_cfg = get_optional(cfg, "paths.final_selected_radiomics", default=None)
    if final_cfg:
        return (
            Path(get_required(cfg, "paths.final_selected_radiomics.development_csv")),
            Path(get_required(cfg, "paths.final_selected_radiomics.internal_test_csv")),
            Path(get_required(cfg, "paths.final_selected_radiomics.external_test_csv")),
        )

    final_root = Path(get_required(cfg, "paths.selected_features_root")) / "final"
    return (
        final_root / "development_selected.csv",
        final_root / "internal_test_selected.csv",
        final_root / "external_test_selected.csv",
    )


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


def _build_base_transform(cfg: Mapping[str, Any], *, training: bool) -> Compose:
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
            transforms.append(
                RandFlipd(
                    keys=["image", "mask"],
                    prob=float(aug_cfg.get("flip_probability", 0.5)),
                    spatial_axis=[0, 1, 2],
                )
            )
        if bool(aug_cfg.get("enable_random_rotate90", True)):
            transforms.append(
                RandRotate90d(
                    keys=["image", "mask"],
                    prob=float(aug_cfg.get("rotate90_probability", 0.5)),
                    max_k=3,
                    spatial_axes=(0, 1),
                )
            )
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
            "cohort": row["cohort"],
        }

    @staticmethod
    def _clone_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": row["image"].clone(),
            "rad_features": row["rad_features"].clone(),
            "label": row["label"].clone(),
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "cohort": row["cohort"],
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_row = self.cached_rows[index] if index < len(self.cached_rows) else self._prepare_row(self.rows[index])
        return self._clone_row(base_row)


def _align_feature_tables(
    development_table: pd.DataFrame,
    internal_table: pd.DataFrame,
    external_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = list(development_table.columns)
    for table_name, table in [("internal", internal_table), ("external", external_table)]:
        missing = [feat for feat in feature_cols if feat not in table.columns]
        if missing:
            raise RuntimeError(
                f"The {table_name} selected radiomics table is missing required features. Examples: {missing[:10]}"
            )
    return (
        development_table.loc[:, feature_cols].copy(),
        internal_table.loc[:, feature_cols].copy(),
        external_table.loc[:, feature_cols].copy(),
    )


def _scale_feature_tables(
    development_table: pd.DataFrame,
    internal_table: pd.DataFrame,
    external_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    development_scaled = scaler.fit_transform(imputer.fit_transform(development_table))
    internal_scaled = scaler.transform(imputer.transform(internal_table))
    external_scaled = scaler.transform(imputer.transform(external_table))

    return (
        pd.DataFrame(development_scaled, index=development_table.index, columns=development_table.columns),
        pd.DataFrame(internal_scaled, index=internal_table.index, columns=internal_table.columns),
        pd.DataFrame(external_scaled, index=external_table.index, columns=external_table.columns),
    )


def _build_rows(
    rows: pd.DataFrame,
    *,
    sample_id_col: str,
    patient_id_col: Optional[str],
    label_col: str,
    image_dir: Path,
    mask_dir: Path,
    suffixes: List[str],
    radiomics_table: pd.DataFrame,
    cohort_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    dropped = {"missing_radiomics": 0, "missing_image": 0, "missing_mask": 0}

    for _, row in rows.iterrows():
        sample_id = normalize_sample_id(row[sample_id_col])
        if sample_id not in radiomics_table.index:
            dropped["missing_radiomics"] += 1
            continue
        try:
            image_path = resolve_image_path(image_dir, sample_id, suffixes)
        except FileNotFoundError:
            dropped["missing_image"] += 1
            continue
        try:
            mask_path = _resolve_mask_path(mask_dir, sample_id, suffixes)
        except FileNotFoundError:
            dropped["missing_mask"] += 1
            continue

        patient_id = sample_id
        if patient_id_col is not None and patient_id_col in row.index:
            patient_id = str(row[patient_id_col])

        records.append(
            {
                "image": str(image_path),
                "mask": str(mask_path),
                "rad_features": radiomics_table.loc[sample_id].to_numpy(dtype=np.float32, copy=True),
                "label": int(row[label_col]),
                "sample_id": sample_id,
                "patient_id": patient_id,
                "cohort": cohort_name,
            }
        )

    return records, dropped


def build_final_dataloaders(cfg: Mapping[str, Any]) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    external_patient_id_col = get_optional(cfg, "columns.external_patient_id", default=None)

    internal_meta = _read_internal_metadata(cfg)
    external_meta = _read_external_metadata(cfg)
    development_group_value = str(get_required(cfg, "dataset_roles.development_group_value"))
    internal_test_group_value = str(get_required(cfg, "dataset_roles.internal_test_group_value"))

    development_rows = internal_meta.loc[internal_meta[group_col] == development_group_value].copy()
    internal_test_rows = internal_meta.loc[internal_meta[group_col] == internal_test_group_value].copy()
    external_rows = external_meta.copy()

    if development_rows.empty:
        raise RuntimeError("The development cohort is empty after filtering the internal metadata file.")

    dev_selected_path, internal_selected_path, external_selected_path = _resolve_final_selected_paths(cfg)
    development_table = _read_selected_table(dev_selected_path)
    internal_table = _read_selected_table(internal_selected_path)
    external_table = _read_selected_table(external_selected_path)
    development_table, internal_table, external_table = _align_feature_tables(
        development_table,
        internal_table,
        external_table,
    )
    development_table, internal_table, external_table = _scale_feature_tables(
        development_table,
        internal_table,
        external_table,
    )

    suffixes = [str(v) for v in get_required(cfg, "dataset.allowed_image_suffixes")]
    development_records, development_dropped = _build_rows(
        development_rows,
        sample_id_col=sample_id_col,
        patient_id_col=patient_id_col,
        label_col=label_col,
        image_dir=Path(get_required(cfg, "paths.images.development_dir")),
        mask_dir=Path(get_required(cfg, "paths.masks.development_dir")),
        suffixes=suffixes,
        radiomics_table=development_table,
        cohort_name="development",
    )
    internal_records, internal_dropped = _build_rows(
        internal_test_rows,
        sample_id_col=sample_id_col,
        patient_id_col=patient_id_col,
        label_col=label_col,
        image_dir=Path(get_required(cfg, "paths.images.internal_test_dir")),
        mask_dir=Path(get_required(cfg, "paths.masks.internal_test_dir")),
        suffixes=suffixes,
        radiomics_table=internal_table,
        cohort_name="internal_test",
    )
    external_records, external_dropped = _build_rows(
        external_rows,
        sample_id_col=sample_id_col,
        patient_id_col=str(external_patient_id_col) if external_patient_id_col is not None else None,
        label_col=label_col,
        image_dir=Path(get_required(cfg, "paths.images.external_test_dir")),
        mask_dir=Path(get_required(cfg, "paths.masks.external_test_dir")),
        suffixes=suffixes,
        radiomics_table=external_table,
        cohort_name="external_test",
    )

    if not development_records:
        raise RuntimeError("The final development training set is empty after intersecting metadata, images, masks, and radiomics.")

    use_cache = bool(get_optional(cfg, "runtime.use_cache_dataset", default=True))
    cache_rate = float(get_optional(cfg, "runtime.cache_rate", default=1.0)) if use_cache else 0.0
    batch_size = int(get_required(cfg, "runtime.batch_size"))
    eval_batch_size = int(get_optional(cfg, "runtime.eval_batch_size", default=batch_size))
    num_workers = int(get_optional(cfg, "runtime.num_workers", default=0))
    pin_memory = bool(get_optional(cfg, "runtime.pin_memory", default=True))
    seed = int(get_required(cfg, "cross_validation.seed"))

    train_dataset = _FusionDataset(
        development_records,
        base_transform=_build_base_transform(cfg, training=True),
        cache_rate=cache_rate,
    )
    internal_dataset = _FusionDataset(
        internal_records,
        base_transform=_build_base_transform(cfg, training=False),
        cache_rate=cache_rate,
    )
    external_dataset = _FusionDataset(
        external_records,
        base_transform=_build_base_transform(cfg, training=False),
        cache_rate=cache_rate,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    internal_loader = DataLoader(
        internal_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    external_loader = DataLoader(
        external_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    log.info(
        "Final dataloaders ready | development=%d internal=%d external=%d",
        len(development_records),
        len(internal_records),
        len(external_records),
    )

    metadata = {
        "development_n": int(len(development_records)),
        "internal_test_n": int(len(internal_records)),
        "external_test_n": int(len(external_records)),
        "radiomics_input_dim": int(development_table.shape[1]),
        "dropped_rows": {
            "development": development_dropped,
            "internal_test": internal_dropped,
            "external_test": external_dropped,
        },
    }
    return {
        "train": train_loader,
        "internal_test": internal_loader,
        "external_test": external_loader,
    }, metadata


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    configure_logging()
    _, metadata = build_final_dataloaders(cfg)
    log.info("Final dataloader summary: %s", metadata)


if __name__ == "__main__":
    main()
