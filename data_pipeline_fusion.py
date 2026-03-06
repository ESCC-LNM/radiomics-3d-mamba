from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fusion_data_pipeline_v2")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-safe fusion dataloaders for 5-fold tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_path = parser.add_argument_group("Paths")
    g_path.add_argument("--internal_csv", type=Path, default=Path("./data3D/internal_group.csv"))
    g_path.add_argument("--external_csv", type=Path, default=Path("./data3D/external_label.csv"))
    g_path.add_argument("--trainval_img_dir", type=Path, default=Path("./data3D/trainval_images"))
    g_path.add_argument("--internal_test_img_dir", type=Path, default=Path("./data3D/internal_test_images"))
    g_path.add_argument("--external_test_img_dir", type=Path, default=Path("./data3D/external_test_images"))
    g_path.add_argument("--rad_trainval_csv", type=Path, default=Path("./outputs/selected_features/fold01/radiomics_internal_trainval_sel.csv"))
    g_path.add_argument("--rad_internal_test_csv", type=Path, default=Path("./outputs/selected_features/fold01/radiomics_internal_test_sel.csv"))
    g_path.add_argument("--rad_external_test_csv", type=Path, default=Path("./outputs/selected_features/fold01/radiomics_external_test_sel.csv"))
    g_path.add_argument("--split_manifest_csv", type=Path, required=True)

    g_schema = parser.add_argument_group("Metadata Schema")
    g_schema.add_argument("--id_col", type=str, default="ID")
    g_schema.add_argument("--label_col", type=str, default="label")
    g_schema.add_argument("--group_col", type=str, default="group")
    g_schema.add_argument("--patient_id_col", type=str, default="patient_id")
    g_schema.add_argument("--train_group_value", type=str, default="train")
    g_schema.add_argument("--internal_test_group_value", type=str, default="test")

    g_io = parser.add_argument_group("I/O")
    g_io.add_argument("--nifti_exts", type=str, nargs="+", default=[".nii.gz", ".nii", ".nii(1).gz"])
    g_io.add_argument("--return_ids", action="store_true")
    g_io.add_argument("--anonymize_ids", action="store_true")

    g_cv = parser.add_argument_group("Fold")
    g_cv.add_argument("--fold_idx", type=int, default=0, help="0-based fold index")
    g_cv.add_argument("--seed", type=int, default=42)

    g_hp = parser.add_argument_group("Transforms")
    g_hp.add_argument("--batch_size", type=int, default=16)
    g_hp.add_argument("--num_workers", type=int, default=4)
    g_hp.add_argument("--roi_size", type=int, nargs=3, default=[32, 64, 64])
    g_hp.add_argument("--window_width", type=float, default=400.0)
    g_hp.add_argument("--window_level", type=float, default=40.0)
    g_hp.add_argument("--no_cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_id(raw_id: Any) -> str:
    s = str(raw_id).strip()
    for suf in (".nii.gz", ".nii(1).gz", ".nii"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s


def anonymize_id(s: str) -> str:
    salt = os.environ.get("FUSION_ID_SALT", "")
    digest = hashlib.sha256((salt + str(s)).encode("utf-8")).hexdigest()[:12]
    return f"case_{digest}"


def find_nifti_by_id(directory: Path, file_id: str, exts: Sequence[str]) -> Optional[Path]:
    fid = str(file_id).strip()
    for ext in exts:
        p = directory / f"{fid}{ext}"
        if p.exists():
            return p
    return None


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str, "patient_id": str})
    for c in ("ID", "label", "patient_id", "fold"):
        if c not in df.columns:
            raise ValueError(f"Split manifest missing required column '{c}': {path}")
    df["ID"] = df["ID"].astype(str).map(clean_id)
    return df


def read_metadata(path: Path, id_col: str, label_col: str, group_col: str, patient_id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={id_col: str, patient_id_col: str})
    for c in (id_col, label_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Metadata CSV missing required column '{c}': {path}")
    df[id_col] = df[id_col].astype(str).map(clean_id)
    if patient_id_col in df.columns:
        df[patient_id_col] = df[patient_id_col].astype(str).str.strip()
    else:
        df[patient_id_col] = df[id_col].astype(str)
    return df


def read_radiomics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).map(clean_id)
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].tolist()[:10]
        raise ValueError(f"Duplicate IDs in radiomics CSV {path}: {dup}")
    df = df.set_index("ID")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        raise ValueError(f"No radiomics features found in {path}")
    return df


def maybe_attach_id(item: Dict[str, Any], raw_id: str, return_ids: bool, anonymize_flag: bool) -> None:
    if not return_ids:
        return
    item["id"] = anonymize_id(raw_id) if anonymize_flag else raw_id


# -----------------------------------------------------------------------------
# Items
# -----------------------------------------------------------------------------
def create_trainval_items(
    internal_csv: Path,
    manifest_csv: Path,
    trainval_img_dir: Path,
    rad_train_df: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    group_col: str,
    patient_id_col: str,
    train_group_value: str,
    nifti_exts: Sequence[str],
    return_ids: bool,
    anonymize_ids_flag: bool,
) -> List[Dict[str, Any]]:
    df_internal = read_metadata(internal_csv, id_col, label_col, group_col, patient_id_col)
    manifest = load_manifest(manifest_csv)
    manifest_map = manifest.set_index("ID")

    df_train = df_internal[df_internal[group_col].astype(str).str.strip() == str(train_group_value)].copy()

    items: List[Dict[str, Any]] = []
    dropped_missing_manifest = 0
    dropped_missing_img = 0
    dropped_missing_rad = 0

    for _, row in df_train.iterrows():
        sid = clean_id(row[id_col])
        if sid not in manifest_map.index:
            dropped_missing_manifest += 1
            continue
        img_path = find_nifti_by_id(Path(trainval_img_dir), sid, nifti_exts)
        if img_path is None:
            dropped_missing_img += 1
            continue
        if sid not in rad_train_df.index:
            dropped_missing_rad += 1
            continue

        meta = manifest_map.loc[sid]
        item: Dict[str, Any] = {
            "image": img_path,
            "label": torch.tensor(int(meta["label"]), dtype=torch.long),
            "rad_id": sid,
            "patient_id": str(meta["patient_id"]),
            "fold": int(meta["fold"]),
        }
        maybe_attach_id(item, raw_id=str(row[id_col]).strip(), return_ids=return_ids, anonymize_flag=anonymize_ids_flag)
        items.append(item)

    if dropped_missing_manifest or dropped_missing_img or dropped_missing_rad:
        log.warning(
            "Dropped trainval samples | no_manifest=%d | no_image=%d | no_radiomics=%d",
            dropped_missing_manifest,
            dropped_missing_img,
            dropped_missing_rad,
        )
    if not items:
        raise RuntimeError("No train/val items after matching manifest, image files, and radiomics rows.")
    return items


def create_test_items(
    csv_path: Path,
    image_dir: Path,
    rad_df: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    group_col: Optional[str],
    group_value: Optional[str],
    patient_id_col: str,
    nifti_exts: Sequence[str],
    return_ids: bool,
    anonymize_ids_flag: bool,
) -> List[Dict[str, Any]]:
    df = read_metadata(csv_path, id_col, label_col, group_col if group_col is not None else label_col, patient_id_col)
    if group_col is not None and group_value is not None:
        df = df[df[group_col].astype(str).str.strip() == str(group_value)].copy()

    items: List[Dict[str, Any]] = []
    dropped_missing_img = 0
    dropped_missing_rad = 0
    for _, row in df.iterrows():
        sid = clean_id(row[id_col])
        img_path = find_nifti_by_id(Path(image_dir), sid, nifti_exts)
        if img_path is None:
            dropped_missing_img += 1
            continue
        if sid not in rad_df.index:
            dropped_missing_rad += 1
            continue
        item: Dict[str, Any] = {
            "image": img_path,
            "label": torch.tensor(int(row[label_col]), dtype=torch.long),
            "rad_id": sid,
            "patient_id": str(row.get(patient_id_col, sid)),
        }
        maybe_attach_id(item, raw_id=str(row[id_col]).strip(), return_ids=return_ids, anonymize_flag=anonymize_ids_flag)
        items.append(item)
    if dropped_missing_img or dropped_missing_rad:
        log.warning(
            "Dropped test samples from %s | no_image=%d | no_radiomics=%d",
            csv_path.name,
            dropped_missing_img,
            dropped_missing_rad,
        )
    return items


# -----------------------------------------------------------------------------
# Radiomics preprocess (fit on fold train only)
# -----------------------------------------------------------------------------
def _stack_feats(items: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack([it["rad_features"].detach().cpu().numpy() for it in items], axis=0)


def _write_feats(items: List[Dict[str, Any]], feats: np.ndarray) -> None:
    for i, it in enumerate(items):
        it["rad_features"] = torch.tensor(feats[i], dtype=torch.float32)


def attach_raw_radiomics(items: List[Dict[str, Any]], rad_df: pd.DataFrame) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    missing = 0
    for it in items:
        rid = it["rad_id"]
        if rid not in rad_df.index:
            missing += 1
            continue
        feats = rad_df.loc[rid].values.astype(np.float32, copy=False)
        it["rad_features"] = torch.tensor(feats, dtype=torch.float32)
        kept.append(it)
    if missing > 0:
        log.warning("Dropped %d samples due to missing raw radiomics rows.", missing)
    return kept


def apply_fold_isolated_preprocess(
    fold_train: List[Dict[str, Any]],
    fold_val: List[Dict[str, Any]],
    int_test: List[Dict[str, Any]],
    ext_test: List[Dict[str, Any]],
) -> None:
    X_tr = _stack_feats(fold_train)
    X_va = _stack_feats(fold_val) if fold_val else None
    X_it = _stack_feats(int_test) if int_test else None
    X_et = _stack_feats(ext_test) if ext_test else None

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    if X_va is not None:
        X_va = imputer.transform(X_va)
    if X_it is not None:
        X_it = imputer.transform(X_it)
    if X_et is not None:
        X_et = imputer.transform(X_et)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    if X_va is not None:
        X_va = scaler.transform(X_va)
    if X_it is not None:
        X_it = scaler.transform(X_it)
    if X_et is not None:
        X_et = scaler.transform(X_et)

    _write_feats(fold_train, X_tr)
    if X_va is not None:
        _write_feats(fold_val, X_va)
    if X_it is not None:
        _write_feats(int_test, X_it)
    if X_et is not None:
        _write_feats(ext_test, X_et)


# -----------------------------------------------------------------------------
# MONAI transforms
# -----------------------------------------------------------------------------
def get_train_transforms(roi_size: Tuple[int, int, int], window_min: float, window_max: float) -> Compose:
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(keys="image", a_min=window_min, a_max=window_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
        RandFlipd(keys="image", spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys="image", prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys="image",
            spatial_size=roi_size,
            prob=0.3,
            rotate_range=(0.2, 0.2, 0.2),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros",
        ),
        ToTensord(keys=["image"]),
    ])


def get_eval_transforms(roi_size: Tuple[int, int, int], window_min: float, window_max: float) -> Compose:
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(keys="image", a_min=window_min, a_max=window_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
        ToTensord(keys=["image"]),
    ])


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def peek_radiomics_dim(rad_csv: Path) -> int:
    df = pd.read_csv(rad_csv, nrows=1)
    cols = list(df.columns)
    if "ID" in cols:
        return len([c for c in cols if c != "ID"])
    return max(0, len(cols) - 1)


def get_fusion_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    roi_size = tuple(int(x) for x in args.roi_size)
    window_min = float(args.window_level) - float(args.window_width) / 2.0
    window_max = float(args.window_level) + float(args.window_width) / 2.0

    rad_train_df = read_radiomics_csv(Path(args.rad_trainval_csv))
    rad_int_df = read_radiomics_csv(Path(args.rad_internal_test_csv))
    rad_ext_df = read_radiomics_csv(Path(args.rad_external_test_csv))

    trainval_items = create_trainval_items(
        internal_csv=Path(args.internal_csv),
        manifest_csv=Path(args.split_manifest_csv),
        trainval_img_dir=Path(args.trainval_img_dir),
        rad_train_df=rad_train_df,
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=args.group_col,
        patient_id_col=args.patient_id_col,
        train_group_value=args.train_group_value,
        nifti_exts=args.nifti_exts,
        return_ids=bool(args.return_ids),
        anonymize_ids_flag=bool(args.anonymize_ids),
    )
    int_test_items = create_test_items(
        csv_path=Path(args.internal_csv),
        image_dir=Path(args.internal_test_img_dir),
        rad_df=rad_int_df,
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=args.group_col,
        group_value=args.internal_test_group_value,
        patient_id_col=args.patient_id_col,
        nifti_exts=args.nifti_exts,
        return_ids=bool(args.return_ids),
        anonymize_ids_flag=bool(args.anonymize_ids),
    )
    ext_test_items = create_test_items(
        csv_path=Path(args.external_csv),
        image_dir=Path(args.external_test_img_dir),
        rad_df=rad_ext_df,
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=None,
        group_value=None,
        patient_id_col=args.patient_id_col,
        nifti_exts=args.nifti_exts,
        return_ids=bool(args.return_ids),
        anonymize_ids_flag=bool(args.anonymize_ids),
    )

    trainval_items = attach_raw_radiomics(trainval_items, rad_train_df)
    int_test_items = attach_raw_radiomics(int_test_items, rad_int_df)
    ext_test_items = attach_raw_radiomics(ext_test_items, rad_ext_df)

    fold_idx = int(args.fold_idx)
    fold_train = [it for it in trainval_items if int(it["fold"]) != fold_idx]
    fold_val = [it for it in trainval_items if int(it["fold"]) == fold_idx]
    if not fold_train or not fold_val:
        raise RuntimeError(f"Fold {fold_idx} produced empty train or val set.")

    apply_fold_isolated_preprocess(fold_train, fold_val, int_test_items, ext_test_items)

    train_tf = get_train_transforms(roi_size, window_min, window_max)
    eval_tf = get_eval_transforms(roi_size, window_min, window_max)

    dataset_type = CacheDataset if bool(getattr(args, "use_cache", True)) else Dataset
    cache_kwargs = {"cache_rate": 1.0, "num_workers": int(args.num_workers)} if bool(getattr(args, "use_cache", True)) else {}
    nw = min(2, int(args.num_workers)) if bool(getattr(args, "use_cache", True)) else int(args.num_workers)

    train_ds = dataset_type(data=fold_train, transform=train_tf, **cache_kwargs)
    val_ds = dataset_type(data=fold_val, transform=eval_tf, **cache_kwargs)
    int_ds = dataset_type(data=int_test_items, transform=eval_tf, **cache_kwargs)
    ext_ds = dataset_type(data=ext_test_items, transform=eval_tf, **cache_kwargs)

    g = torch.Generator()
    g.manual_seed(int(getattr(args, "seed", 42)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=nw,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    int_loader = DataLoader(
        int_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    ext_loader = DataLoader(
        ext_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )

    log.info(
        "Fold %d dataloaders ready | train=%d | val=%d | internal_test=%d | external=%d | rad_dim=%d",
        fold_idx,
        len(fold_train),
        len(fold_val),
        len(int_test_items),
        len(ext_test_items),
        rad_train_df.shape[1],
    )
    return {"train": train_loader, "val": val_loader, "internal_test": int_loader, "external": ext_loader}


if __name__ == "__main__":
    args = get_args()
    loaders = get_fusion_dataloaders(args)
    for name, loader in loaders.items():
        print(name, len(loader))
