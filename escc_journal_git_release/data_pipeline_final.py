"""
Final Global Fusion Data Pipeline (3D NIfTI + Radiomics CSV)

Methodological Integrity:
1. Global Training Pooling: All 'train' group samples are aggregated into a single training loader. No validation split.
2. Strict Prospective Scaling: Radiomics median imputation and StandardScaler are fitted EXCLUSIVELY on the global training set.
3. Zero Data Leakage: The fitted statistical parameters are unidirectionally applied to internal and external test sets.
"""

import os
import sys
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np
import torch

# MONAI
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRanged,
    ResizeD,
    RandFlipD,
    RandRotate90D,
    RandAffineD,
    ToTensorD,
)
from monai.utils import set_determinism

# Scikit-learn
from sklearn.preprocessing import StandardScaler

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DataPipelineFinal")


# ====================================================================
# A. Argument Parsing (for standalone testing)
# ====================================================================
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Global Fusion Data Pipeline (3D Image + Radiomics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    DEV_BASE_DIR = Path("/root/autodl-tmp/data")
    DEV_RADIOMICS_DIR = DEV_BASE_DIR / "radiomics_features"

    g_path = parser.add_argument_group("Path Configuration")
    g_path.add_argument("--base_dir", type=Path, default=DEV_BASE_DIR)
    g_path.add_argument("--radiomics_dir", type=Path, default=DEV_RADIOMICS_DIR)

    g_path.add_argument("--internal_csv", type=Path, default=DEV_BASE_DIR / "internal group.csv")
    g_path.add_argument("--external_csv", type=Path, default=DEV_BASE_DIR / "external label.csv")

    g_path.add_argument("--trainval_img_dir", type=Path, default=DEV_BASE_DIR / "trainandval crop 3d")
    g_path.add_argument("--internal_test_img_dir", type=Path, default=DEV_BASE_DIR / "internal test crop 3d")
    g_path.add_argument("--external_test_img_dir", type=Path, default=DEV_BASE_DIR / "external LN 294 crop 3d")
    g_path.add_argument("--trainval_mask_dir", type=Path, default=DEV_BASE_DIR / "trainandval masks")
    g_path.add_argument("--internal_test_mask_dir", type=Path, default=DEV_BASE_DIR / "internal test masks")
    g_path.add_argument("--external_test_mask_dir", type=Path, default=DEV_BASE_DIR / "external test masks")

    g_path.add_argument("--rad_trainval_csv", type=Path,
                        default=DEV_RADIOMICS_DIR / "radiomics_internal_trainval_sel.csv")
    g_path.add_argument("--rad_internal_test_csv", type=Path,
                        default=DEV_RADIOMICS_DIR / "radiomics_internal_test_sel.csv")
    g_path.add_argument("--rad_external_test_csv", type=Path,
                        default=DEV_RADIOMICS_DIR / "radiomics_external_test_sel.csv")

    g_hp = parser.add_argument_group("Hyperparameters & Settings")
    g_hp.add_argument("--batch_size", type=int, default=16)
    g_hp.add_argument("--num_workers", type=int, default=4)
    g_hp.add_argument("--roi_size", type=int, nargs=3, default=[32, 64, 64])
    g_hp.add_argument("--window_width", type=float, default=400.0)
    g_hp.add_argument("--window_level", type=float, default=40.0)
    g_hp.add_argument("--seed", type=int, default=42, help="Repro seed for transforms and dataloader workers")

    g_hp.add_argument("--no_cache", dest="use_cache", action="store_false", help="Disable CacheDataset")
    parser.set_defaults(use_cache=True)

    return parser.parse_args()


# ====================================================================
# B. ID & DataFrame Helpers
# ====================================================================
def clean_id_from_csv(raw_id: Any) -> str:
    s = str(raw_id).strip()
    return s.replace(".nii.gz", "").replace(".nii(1).gz", "").replace(".nii", "")


def find_file_by_id(directory: Path, file_id: str) -> Optional[Path]:
    file_id = str(file_id).strip()
    for suffix in [".nii.gz", ".nii(1).gz"]:
        path = directory / f"{file_id}{suffix}"
        if path.exists():
            return path
    return None


def find_mask_by_id(mask_directory: Path, image_directory: Path, file_id: str) -> Optional[Path]:
    """
    Resolve segmentation mask path by ID.

    If mask directory equals image directory, avoid picking the raw image file with plain ID.
    """
    fid = str(file_id).strip()
    suffixes = [".nii.gz", ".nii(1).gz"]

    mask_stems = [
        f"{fid}_mask",
        f"{fid}_seg",
        f"{fid}_label",
        f"{fid}-mask",
        f"{fid}-seg",
        f"{fid}-label",
    ]

    # Allow plain-ID masks only when mask/image folders differ.
    if Path(mask_directory).absolute() != Path(image_directory).absolute():
        mask_stems = [fid] + mask_stems

    for stem in mask_stems:
        for suffix in suffixes:
            p = mask_directory / f"{stem}{suffix}"
            if p.exists():
                return p
    return None


def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ====================================================================
# C. MONAI Transforms
# ====================================================================
def get_train_transforms(roi_size: Tuple[int, int, int], window_min: float, window_max: float) -> Compose:
    return Compose([
        LoadImageD(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image", "mask"]),
        ScaleIntensityRanged(keys="image", a_min=window_min, a_max=window_max, b_min=0.0, b_max=1.0, clip=True),
        ResizeD(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
        ResizeD(keys="mask", spatial_size=roi_size, mode="nearest"),
        RandFlipD(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90D(keys=["image", "mask"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandAffineD(
            keys=["image", "mask"], spatial_size=roi_size, prob=0.3,
            rotate_range=(0.2, 0.2, 0.2), scale_range=(0.1, 0.1, 0.1), padding_mode="zeros",
            mode=["trilinear", "nearest"],
        ),
        ToTensorD(keys=["image", "mask"]),
    ])


def get_test_transforms(roi_size: Tuple[int, int, int], window_min: float, window_max: float) -> Compose:
    return Compose([
        LoadImageD(keys=["image", "mask"]),
        EnsureChannelFirstD(keys=["image", "mask"]),
        ScaleIntensityRanged(keys="image", a_min=window_min, a_max=window_max, b_min=0.0, b_max=1.0, clip=True),
        ResizeD(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
        ResizeD(keys="mask", spatial_size=roi_size, mode="nearest"),
        ToTensorD(keys=["image", "mask"]),
    ])


# ====================================================================
# D. Radiomics Processing (Strict Isolation)
# ====================================================================
def load_radiomics_raw(
        rad_train_csv: Path, rad_int_test_csv: Path, rad_ext_test_csv: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("--- Loading Radiomics (RAW; Pre-scaling) ---")
    df_train = pd.read_csv(rad_train_csv, dtype={"ID": str}).set_index("ID")
    df_int_test = pd.read_csv(rad_int_test_csv, dtype={"ID": str}).set_index("ID")
    df_ext_test = pd.read_csv(rad_ext_test_csv, dtype={"ID": str}).set_index("ID")

    df_train.index = df_train.index.map(clean_id_from_csv)
    df_int_test.index = df_int_test.index.map(clean_id_from_csv)
    df_ext_test.index = df_ext_test.index.map(clean_id_from_csv)

    df_train = _ensure_numeric_df(df_train)
    df_int_test = _ensure_numeric_df(df_int_test)
    df_ext_test = _ensure_numeric_df(df_ext_test)

    feature_cols = list(df_train.columns)
    df_int_test = df_int_test.reindex(columns=feature_cols)
    df_ext_test = df_ext_test.reindex(columns=feature_cols)

    log.info(f"  Radiomics feature count (canonical columns): {len(feature_cols)}")
    return df_train, df_int_test, df_ext_test


def fit_global_radiomics_preprocessor(
        df_train_raw: pd.DataFrame, global_train_ids: List[str]
) -> Tuple[pd.Series, StandardScaler]:
    """
    Fits median imputer and StandardScaler ONLY on the global training set IDs.
    This guarantees zero leakage to the internal/external test sets.
    """
    # Filter to exact IDs actually present in the image train dataset
    train_mat = df_train_raw.loc[df_train_raw.index.intersection(global_train_ids)].copy()

    med = train_mat.median(numeric_only=True).fillna(0.0)
    train_filled = train_mat.fillna(med).fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(train_filled.values)

    return med, scaler


def apply_radiomics_preprocessor(
        df_raw: pd.DataFrame, med: pd.Series, scaler: StandardScaler
) -> pd.DataFrame:
    df = df_raw.copy()
    med_aligned = med.reindex(df.columns).fillna(0.0)
    x = df.fillna(med_aligned).fillna(0.0).values
    df.loc[:, :] = scaler.transform(x)
    return df


def attach_radiomics_to_items(items: List[Dict[str, Any]], df_scaled: pd.DataFrame) -> List[Dict[str, Any]]:
    kept = []
    missing = 0
    for it in items:
        rid = it.get("rad_id")
        if rid is None or rid not in df_scaled.index:
            missing += 1
            continue
        feats = df_scaled.loc[rid].values.astype(np.float32, copy=False)
        it["rad_features"] = torch.tensor(feats, dtype=torch.float32)
        kept.append(it)

    if missing > 0:
        log.warning(f"  Dropped {missing} samples due to missing radiomics rows.")
    return kept


# ====================================================================
# E. Data List Aggregation
# ====================================================================
def create_global_fusion_lists(
        internal_csv: Path, external_csv: Path,
        train_img_dir: Path, int_test_img_dir: Path, ext_test_img_dir: Path,
        train_mask_dir: Path, int_test_mask_dir: Path, ext_test_mask_dir: Path,
        df_rad_train_raw: pd.DataFrame, df_rad_int_raw: pd.DataFrame, df_rad_ext_raw: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    log.info("--- Aggregating Global Sample Lists (No K-Fold Splitting; strict image+mask+radiomics intersection) ---")
    df_internal = pd.read_csv(internal_csv)
    df_external = pd.read_csv(external_csv)

    global_train_files, internal_test_files, external_test_files = [], [], []
    stats = {
        "missing_rad_train": 0,
        "missing_rad_int": 0,
        "missing_rad_ext": 0,
        "missing_img_train": 0,
        "missing_img_int": 0,
        "missing_img_ext": 0,
        "missing_mask_train": 0,
        "missing_mask_int": 0,
        "missing_mask_ext": 0,
    }

    for _, row in df_internal.iterrows():
        clean_id = clean_id_from_csv(row["ID"])
        original_id = str(row["ID"]).strip()
        label = int(row["label"])

        if row["group"] == "train":
            base_dir, rad_df_raw, dst = train_img_dir, df_rad_train_raw, global_train_files
            miss_rad_key, miss_img_key = "missing_rad_train", "missing_img_train"
            mask_dir, miss_mask_key = train_mask_dir, "missing_mask_train"
        elif row["group"] == "test":
            base_dir, rad_df_raw, dst = int_test_img_dir, df_rad_int_raw, internal_test_files
            miss_rad_key, miss_img_key = "missing_rad_int", "missing_img_int"
            mask_dir, miss_mask_key = int_test_mask_dir, "missing_mask_int"
        else:
            continue

        if clean_id not in rad_df_raw.index:
            stats[miss_rad_key] += 1
            continue

        img_path = find_file_by_id(base_dir, clean_id)
        if img_path is None:
            stats[miss_img_key] += 1
            continue

        mask_path = find_mask_by_id(mask_dir, base_dir, clean_id)
        if mask_path is None:
            stats[miss_mask_key] += 1
            continue

        dst.append(
            {
                "image": img_path,
                "mask": mask_path,
                "label": torch.tensor(label, dtype=torch.long),
                "id": original_id,
                "rad_id": clean_id,
            }
        )

    for _, row in df_external.iterrows():
        clean_id = clean_id_from_csv(row["ID"])
        original_id = str(row["ID"]).strip()
        label = int(row["label"])

        if clean_id not in df_rad_ext_raw.index:
            stats["missing_rad_ext"] += 1
            continue

        img_path = find_file_by_id(ext_test_img_dir, clean_id)
        if img_path is None:
            stats["missing_img_ext"] += 1
            continue

        mask_path = find_mask_by_id(ext_test_mask_dir, ext_test_img_dir, clean_id)
        if mask_path is None:
            stats["missing_mask_ext"] += 1
            continue

        external_test_files.append(
            {
                "image": img_path,
                "mask": mask_path,
                "label": torch.tensor(label, dtype=torch.long),
                "id": original_id,
                "rad_id": clean_id,
            }
        )

    log.info(f"  Global Train Pool: {len(global_train_files)}")
    log.info(f"  Internal Test:     {len(internal_test_files)}")
    log.info(f"  External Test:     {len(external_test_files)}")
    log.info(
        "  Missing counts | "
        f"RAD(train/int/ext)=({stats['missing_rad_train']}/{stats['missing_rad_int']}/{stats['missing_rad_ext']}), "
        f"IMG(train/int/ext)=({stats['missing_img_train']}/{stats['missing_img_int']}/{stats['missing_img_ext']}), "
        f"MASK(train/int/ext)=({stats['missing_mask_train']}/{stats['missing_mask_int']}/{stats['missing_mask_ext']})"
    )

    if not global_train_files:
        log.error("Critical: Global training set is empty. Check paths.")
        sys.exit(1)

    return global_train_files, internal_test_files, external_test_files


# ====================================================================
# F. Main Entry: Dataloader Creation
# ====================================================================
def _seed_worker(worker_id: int) -> None:
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def get_final_fusion_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    set_determinism(seed=int(getattr(args, "seed", 42)))

    log.info("--- Constructing Final Global Dataloaders ---")

    roi_size = tuple(args.roi_size)
    window_min = float(args.window_level) - (float(args.window_width) / 2.0)
    window_max = float(args.window_level) + (float(args.window_width) / 2.0)

    # 1. Load Raw
    df_train_raw, df_int_raw, df_ext_raw = load_radiomics_raw(
        args.rad_trainval_csv, args.rad_internal_test_csv, args.rad_external_test_csv
    )

    # 2. Build Lists
    train_files, int_test_files, ext_test_files = create_global_fusion_lists(
        args.internal_csv, args.external_csv,
        args.trainval_img_dir, args.internal_test_img_dir, args.external_test_img_dir,
        args.trainval_mask_dir, args.internal_test_mask_dir, args.external_test_mask_dir,
        df_train_raw, df_int_raw, df_ext_raw
    )

    # 3. Fit Scaler ONLY on Global Train IDs
    train_ids = [it["rad_id"] for it in train_files]
    med, scaler = fit_global_radiomics_preprocessor(df_train_raw, train_ids)

    # 4. Apply Scaler Unidirectionally
    df_train_scaled = apply_radiomics_preprocessor(df_train_raw, med, scaler)
    df_int_scaled = apply_radiomics_preprocessor(df_int_raw, med, scaler)
    df_ext_scaled = apply_radiomics_preprocessor(df_ext_raw, med, scaler)

    # 5. Attach Features
    train_files = attach_radiomics_to_items(train_files, df_train_scaled)
    int_test_files = attach_radiomics_to_items(int_test_files, df_int_scaled)
    ext_test_files = attach_radiomics_to_items(ext_test_files, df_ext_scaled)

    # 6. Transforms
    train_transforms = get_train_transforms(roi_size, window_min, window_max)
    test_transforms = get_test_transforms(roi_size, window_min, window_max)

    # 7. Datasets
    dataset_type = CacheDataset if getattr(args, "use_cache", True) else Dataset
    cache_props = {"cache_rate": 1.0, "num_workers": int(args.num_workers)} if getattr(args, "use_cache", True) else {}
    nw = min(2, int(args.num_workers)) if getattr(args, "use_cache", True) else int(args.num_workers)

    train_ds = dataset_type(data=train_files, transform=train_transforms, **cache_props)
    int_test_ds = dataset_type(data=int_test_files, transform=test_transforms, **cache_props)
    ext_test_ds = dataset_type(data=ext_test_files, transform=test_transforms, **cache_props)

    # 8. DataLoaders
    g = torch.Generator()
    g.manual_seed(int(getattr(args, "seed", 42)))

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=nw,
                              worker_init_fn=_seed_worker, generator=g, pin_memory=torch.cuda.is_available())
    int_test_loader = DataLoader(int_test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=nw,
                                 worker_init_fn=_seed_worker, generator=g, pin_memory=torch.cuda.is_available())
    ext_test_loader = DataLoader(ext_test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=nw,
                                 worker_init_fn=_seed_worker, generator=g, pin_memory=torch.cuda.is_available())

    log.info("--- Global Dataloaders Ready ---")
    return {"train": train_loader, "internal_test": int_test_loader, "external_test": ext_test_loader}
