"""
Fusion Data Pipeline (3D NIfTI + Radiomics CSV)

- Loads multi-modal samples: 3D NIfTI images + radiomics feature vectors.
- Provides stratified K-Fold split for internal train/val.
- Applies strictly fold-isolated preprocessing for radiomics features
  (median imputation + standardization fit ONLY on the training fold),
  to prevent cross-validation leakage.

Release-friendly design:
- No hard-coded private absolute paths by default.
- ID handling is privacy-safe by default (disabled unless explicitly enabled).
- Robust CSV parsing with explicit consistency checks (no silent feature dropping).
"""

from __future__ import annotations

import sys
import os
import random
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None  # type: ignore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# MONAI (use canonical *d transform names for version robustness)
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    ToTensord,
)

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fusion_data_pipeline")


# =====================================================================
# CLI (optional standalone usage)
# =====================================================================
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusion Data Pipeline (3D image + radiomics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_path = parser.add_argument_group("Paths")
    g_path.add_argument("--base_dir", type=Path, default=Path("./data"))
    g_path.add_argument("--radiomics_dir", type=Path, default=Path("./outputs/selected_features"))

    # Metadata CSVs
    g_path.add_argument("--internal_csv", type=Path, default=Path("./data/internal_group.csv"))
    g_path.add_argument("--external_csv", type=Path, default=Path("./data/external_label.csv"))

    # Image directories
    g_path.add_argument("--trainval_img_dir", type=Path, default=Path("./data/trainval_images"))
    g_path.add_argument("--internal_test_img_dir", type=Path, default=Path("./data/internal_test_images"))
    g_path.add_argument("--external_test_img_dir", type=Path, default=Path("./data/external_test_images"))

    # Radiomics CSVs (selected)
    g_path.add_argument(
        "--rad_trainval_csv",
        type=Path,
        default=Path("./outputs/selected_features/global/radiomics_internal_trainval_sel.csv"),
    )
    g_path.add_argument(
        "--rad_internal_test_csv",
        type=Path,
        default=Path("./outputs/selected_features/global/radiomics_internal_test_sel.csv"),
    )
    g_path.add_argument(
        "--rad_external_test_csv",
        type=Path,
        default=Path("./outputs/selected_features/global/radiomics_external_test_sel.csv"),
    )

    g_io = parser.add_argument_group("I/O")
    g_io.add_argument(
        "--nifti_exts",
        type=str,
        nargs="+",
        default=[".nii.gz", ".nii", ".nii(1).gz"],
        help="Candidate NIfTI extensions to try when resolving file by ID.",
    )

    g_hp = parser.add_argument_group("Hyperparameters")
    g_hp.add_argument("--batch_size", type=int, default=4)
    g_hp.add_argument("--num_workers", type=int, default=4)
    g_hp.add_argument("--roi_size", type=int, nargs=3, default=[32, 64, 64], help="Target ROI size (D H W)")
    g_hp.add_argument("--window_width", type=float, default=400.0)
    g_hp.add_argument("--window_level", type=float, default=40.0)

    g_cv = parser.add_argument_group("Cross Validation")
    g_cv.add_argument("--n_splits", type=int, default=5)
    g_cv.add_argument("--fold_idx", type=int, default=0, help="Fold index (0-based)")
    g_cv.add_argument("--seed", type=int, default=42)
    g_cv.add_argument("--group_col", type=str, default="patient_id",
                      help="Optional metadata column used to enforce patient-level grouping in CV. "
                           "If present in internal_csv, GroupKFold/StratifiedGroupKFold will be used.")

    g_cache = parser.add_argument_group("Caching")
    g_cache.add_argument("--no_cache", dest="use_cache", action="store_false", help="Disable MONAI CacheDataset")
    parser.set_defaults(use_cache=True)

    g_priv = parser.add_argument_group("Privacy / Debug")
    g_priv.add_argument(
        "--return_ids",
        action="store_true",
        help="Include sample id in each item. Disabled by default for privacy.",
    )
    g_priv.add_argument(
        "--anonymize_ids",
        action="store_true",
        help="If --return_ids is set, anonymize ids using SHA-256 hash.",
    )
    g_priv.add_argument(
        "--debug_print_batch",
        action="store_true",
        help="Print a small batch in __main__ sanity check (can include ids if enabled).",
    )

    return parser.parse_args()


# =====================================================================
# Utilities
# =====================================================================
def _require_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def clean_id(raw_id: Any) -> str:
    """Normalize an ID string by stripping whitespace and removing known nifti extensions."""
    s = str(raw_id).strip()
    for suf in [".nii.gz", ".nii(1).gz", ".nii"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def anonymize_id(s: str) -> str:
    """
    Stable anonymization for release safety.

    Uses SHA-256 hash; optional salt can be provided via env var:
        export FUSION_ID_SALT="your_salt"
    """
    salt = os.environ.get("FUSION_ID_SALT", "")
    msg = (salt + str(s)).encode("utf-8")
    digest = hashlib.sha256(msg).hexdigest()[:12]
    return f"case_{digest}"


def find_nifti_by_id(directory: Path, file_id: str, exts: Sequence[str]) -> Optional[Path]:
    """Resolve a NIfTI file by trying multiple extensions."""
    fid = str(file_id).strip()
    for ext in exts:
        p = directory / f"{fid}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------
# Reproducibility helpers (DataLoader worker seeding)
# ---------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """Ensure each DataLoader worker has a different, deterministic RNG state."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================================
# Transforms
# =====================================================================
def get_train_transforms(
    roi_size: Tuple[int, int, int],
    ww: float,
    wl: float,
) -> Compose:
    a_min = wl - ww / 2.0
    a_max = wl + ww / 2.0
    return Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
            RandFlipd(keys="image", spatial_axis=[0, 1, 2], prob=0.5),
            RandRotate90d(keys="image", prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandAffined(
                keys="image",
                prob=0.3,
                spatial_size=roi_size,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="zeros",
            ),
            ToTensord(keys=["image"]),
        ]
    )


def get_val_transforms(
    roi_size: Tuple[int, int, int],
    ww: float,
    wl: float,
) -> Compose:
    a_min = wl - ww / 2.0
    a_max = wl + ww / 2.0
    return Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys="image", spatial_size=roi_size, mode="trilinear", align_corners=True),
            ToTensord(keys=["image"]),
        ]
    )


# =====================================================================
# Radiomics CSV handling
# =====================================================================
def _read_radiomics_csv(path: Path) -> pd.DataFrame:
    """
    Read radiomics CSV robustly.

    Supports:
    - A proper 'ID' column.
    - A first column that is actually ID (common after reset_index()).
    """
    _require_exists(path, "Radiomics CSV")
    df = pd.read_csv(path)

    # Resolve ID column
    if "ID" not in df.columns:
        first_col = df.columns[0]
        log.warning(f"[{path.name}] No 'ID' column found. Using first column '{first_col}' as ID.")
        df = df.rename(columns={first_col: "ID"})

    df["ID"] = df["ID"].astype(str).map(clean_id)

    # detect duplicates before indexing
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].astype(str).unique().tolist()
        raise ValueError(f"[{path.name}] Duplicate IDs detected (showing up to 10): {dup[:10]}")

    df = df.set_index("ID")

    # Keep only numeric feature columns
    feat_df = df.apply(pd.to_numeric, errors="coerce")
    if feat_df.shape[1] == 0:
        raise ValueError(f"[{path.name}] No numeric radiomics feature columns found.")

    return feat_df


def load_radiomics_tables(
    trainval_csv: Path,
    int_test_csv: Path,
    ext_test_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load radiomics tables WITHOUT fitting any preprocessing statistics.
    Preprocessing (imputation/scaling) is done strictly inside each fold.

    IMPORTANT: This function enforces column consistency across splits to avoid
    silent feature dropping (which can cause dimension mismatch or biased comparisons).
    """
    log.info("--- Loading Radiomics Tables (no preprocessing fit) ---")
    df_trainval = _read_radiomics_csv(trainval_csv)
    df_int_test = _read_radiomics_csv(int_test_csv)
    df_ext_test = _read_radiomics_csv(ext_test_csv)

    cols_tr = set(df_trainval.columns)
    cols_it = set(df_int_test.columns)
    cols_et = set(df_ext_test.columns)

    common_cols = sorted(list(cols_tr & cols_it & cols_et))
    if not common_cols:
        raise ValueError("No common radiomics feature columns across trainval/int_test/ext_test CSVs.")

    # Explicit mismatch reporting (prevents silent dimension changes)
    if len(common_cols) != len(cols_tr) or len(common_cols) != len(cols_it) or len(common_cols) != len(cols_et):
        missing_tr = sorted(list((cols_it & cols_et) - cols_tr))
        missing_it = sorted(list((cols_tr & cols_et) - cols_it))
        missing_et = sorted(list((cols_tr & cols_it) - cols_et))

        msg = (
            "Radiomics columns mismatch detected.\n"
            f"  trainval_cols={len(cols_tr)}, int_test_cols={len(cols_it)}, ext_test_cols={len(cols_et)}\n"
            f"  common_cols={len(common_cols)}\n"
        )
        if missing_tr:
            msg += f"  Missing in trainval (showing up to 10): {missing_tr[:10]}\n"
        if missing_it:
            msg += f"  Missing in internal_test (showing up to 10): {missing_it[:10]}\n"
        if missing_et:
            msg += f"  Missing in external_test (showing up to 10): {missing_et[:10]}\n"
        raise ValueError(msg)

    df_trainval = df_trainval[common_cols]
    df_int_test = df_int_test[common_cols]
    df_ext_test = df_ext_test[common_cols]

    log.info(f"  Radiomics loaded. Features: {len(common_cols)} (consistent across splits)")
    return df_trainval, df_int_test, df_ext_test


# =====================================================================
# Sample list building
# =====================================================================
def create_fusion_lists(
    internal_csv: Path,
    external_csv: Path,
    trainval_img_dir: Path,
    internal_test_img_dir: Path,
    external_test_img_dir: Path,
    rad_tables: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    nifti_exts: Sequence[str],
    return_ids: bool,
    anonymize_ids_flag: bool,
    patient_group_col: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Match metadata rows to image files and radiomics features by cleaned ID.

    Internal CSV MUST provide a 'group' column (train/test) to prevent leakage.
    """
    _require_exists(internal_csv, "Internal metadata CSV")
    _require_exists(external_csv, "External metadata CSV")
    _require_exists(trainval_img_dir, "Train/Val image directory")
    _require_exists(internal_test_img_dir, "Internal test image directory")
    _require_exists(external_test_img_dir, "External test image directory")

    df_internal = pd.read_csv(internal_csv)
    df_external = pd.read_csv(external_csv)

    for req_col in ["ID", "label"]:
        if req_col not in df_internal.columns:
            raise ValueError(f"Internal metadata CSV missing required column: '{req_col}'")
        if req_col not in df_external.columns:
            raise ValueError(f"External metadata CSV missing required column: '{req_col}'")

    # Hard check to prevent silent leakage
    if "group" not in df_internal.columns:
        raise ValueError(
            "Internal metadata CSV must contain a 'group' column with values like 'train'/'test'. "
            "Refusing to proceed to prevent potential leakage."
        )

    df_rad_trainval, df_rad_int_test, df_rad_ext_test = rad_tables

    def _pack_id(raw: Any) -> Optional[str]:
        if not return_ids:
            return None
        s = str(raw).strip()
        return anonymize_id(s) if anonymize_ids_flag else s

    def _pack_group(raw: Any) -> Optional[str]:
        """Pack grouping key (e.g., patient_id) for GroupKFold. Always anonymized if anonymize_ids_flag is True."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        s = str(raw).strip()
        if s == "":
            return None
        return anonymize_id(s) if anonymize_ids_flag else s


    def _process_rows(
        df_meta: pd.DataFrame,
        img_dir: Path,
        rad_df: pd.DataFrame,
        group_filter: Optional[str] = None,
        split_col: str = "group",
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n_missing_img = 0
        n_missing_rad = 0

        for _, row in df_meta.iterrows():
            if group_filter is not None:
                if split_col not in df_meta.columns:
                    raise ValueError(
                        f"Expected column '{split_col}' for group filtering, but it was not found. "
                        "Refusing to proceed to prevent data leakage."
                    )
                if str(row.get(split_col)).strip() != group_filter:
                    continue

            cid = clean_id(row["ID"])
            img_path = find_nifti_by_id(img_dir, cid, nifti_exts)

            if img_path is None:
                n_missing_img += 1
                continue
            if cid not in rad_df.index:
                n_missing_rad += 1
                continue

            feats = rad_df.loc[cid].to_numpy(dtype=np.float32, copy=True)

            item: Dict[str, Any] = {
                "image": img_path,
                "rad_features": torch.from_numpy(feats),  # will be imputed/scaled inside fold
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
            }
            sid = _pack_id(row["ID"])
            if sid is not None:
                item["id"] = sid

            # Optional grouping key for patient-level CV
            if patient_group_col and (patient_group_col in row.index):
                gid = _pack_group(row[patient_group_col])
                if gid is not None:
                    item["group_id"] = gid

            out.append(item)

        if n_missing_img > 0 or n_missing_rad > 0:
            log.warning(
                f"[{img_dir.name}] skipped samples: missing_img={n_missing_img}, missing_rad={n_missing_rad}"
            )

        return out

    # Internal: train/val from internal CSV group == "train"
    trainval_files = _process_rows(df_internal, trainval_img_dir, df_rad_trainval, group_filter="train")

    # Internal test: internal CSV group == "test"
    internal_test_files = _process_rows(df_internal, internal_test_img_dir, df_rad_int_test, group_filter="test")

    # External test: external CSV may not have group column -> take all
    external_test_files = _process_rows(df_external, external_test_img_dir, df_rad_ext_test, group_filter=None)

    if not trainval_files:
        raise RuntimeError("No training samples found after matching. Check paths, IDs, and radiomics tables.")

    log.info(
        f"  Matched samples: trainval={len(trainval_files)}, int_test={len(internal_test_files)}, ext_test={len(external_test_files)}"
    )
    return trainval_files, internal_test_files, external_test_files


# =====================================================================
# Fold-isolated preprocessing (impute + scale)
# =====================================================================
def _stack_feats(items: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack([it["rad_features"].detach().cpu().numpy() for it in items], axis=0)


def _write_feats(items: List[Dict[str, Any]], feats: np.ndarray) -> None:
    for i, it in enumerate(items):
        it["rad_features"] = torch.tensor(feats[i], dtype=torch.float32)


def apply_fold_isolated_preprocess(
    fold_train: List[Dict[str, Any]],
    fold_val: List[Dict[str, Any]],
    int_test: List[Dict[str, Any]],
    ext_test: List[Dict[str, Any]],
) -> None:
    """
    Prevent leakage:
    - Fit imputer (median) on fold_train only -> transform all.
    - Fit scaler on imputed fold_train only -> transform all.
    """
    log.info("  Radiomics preprocessing (fit on TRAIN fold -> transform all)...")

    X_tr = _stack_feats(fold_train)
    X_va = _stack_feats(fold_val) if fold_val else None
    X_it = _stack_feats(int_test) if int_test else None
    X_et = _stack_feats(ext_test) if ext_test else None

    # 1) Median imputation (fit only on fold_train)
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    if X_va is not None:
        X_va = imputer.transform(X_va)
    if X_it is not None:
        X_it = imputer.transform(X_it)
    if X_et is not None:
        X_et = imputer.transform(X_et)

    # 2) Standardization (fit only on fold_train)
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

    log.info("  Radiomics preprocessing done.")


# =====================================================================
# Public API
# =====================================================================
def get_fusion_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    """
    Build dataloaders with stratified K-fold splitting and fold-isolated radiomics preprocessing.
    """
    log.info(f"--- Building Fusion DataLoaders (Fold {args.fold_idx + 1}/{args.n_splits}) ---")

    rad_tables = load_radiomics_tables(args.rad_trainval_csv, args.rad_internal_test_csv, args.rad_external_test_csv)

    trainval_files, int_test_files, ext_test_files = create_fusion_lists(
        internal_csv=args.internal_csv,
        external_csv=args.external_csv,
        trainval_img_dir=args.trainval_img_dir,
        internal_test_img_dir=args.internal_test_img_dir,
        external_test_img_dir=args.external_test_img_dir,
        rad_tables=rad_tables,
        nifti_exts=args.nifti_exts,
        return_ids=bool(args.return_ids),
        anonymize_ids_flag=bool(args.anonymize_ids),
        patient_group_col=str(getattr(args, 'group_col', 'patient_id')),
    )

    labels = np.array([int(it["label"].item()) for it in trainval_files], dtype=np.int64)

    # Optional patient-level (grouped) CV split if group_id is available
    groups = np.array([it.get("group_id", None) for it in trainval_files], dtype=object)
    has_groups = np.all(groups != None)

    # Guardrail: grouped split requires enough unique groups
    if has_groups:
        uniq = len(set([g for g in groups.tolist() if g is not None]))
        if uniq < int(args.n_splits):
            logging.getLogger(__name__).warning(
                "group_col='%s' present but only %d unique groups (< n_splits=%d); "
                "falling back to StratifiedKFold (sample-level).",
                getattr(args, "group_col", "patient_id"),
                uniq,
                int(args.n_splits),
            )
            has_groups = False

    if has_groups:
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
        else:
            # Fallback: GroupKFold (no stratification). Still prevents patient-level leakage.
            splitter = GroupKFold(n_splits=args.n_splits)
            splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
        logging.getLogger(__name__).info("Using grouped CV split with group_col='%s' (patient-level isolation).", getattr(args, "group_col", "patient_id"))
    else:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(labels)), labels))
        logging.getLogger(__name__).warning(
            "No group_id found for train/val samples; falling back to StratifiedKFold (sample-level). "
            "To enforce patient-level isolation, add a '%s' column to internal_csv.", getattr(args, "group_col", "patient_id")
        )

    if not (0 <= args.fold_idx < len(splits)):
        raise IndexError(f"fold_idx={args.fold_idx} out of range for n_splits={args.n_splits}")

    train_idx, val_idx = splits[args.fold_idx]
    fold_train = [trainval_files[i] for i in train_idx]
    fold_val = [trainval_files[i] for i in val_idx]

    apply_fold_isolated_preprocess(fold_train, fold_val, int_test_files, ext_test_files)

    roi_size = tuple(map(int, args.roi_size))
    tr_tf = get_train_transforms(roi_size, float(args.window_width), float(args.window_level))
    va_tf = get_val_transforms(roi_size, float(args.window_width), float(args.window_level))

    ds_cls = CacheDataset if args.use_cache else Dataset
    ds_kwargs = {"cache_rate": 1.0, "num_workers": args.num_workers} if args.use_cache else {}

    if args.use_cache:
        log.info("  Using CacheDataset (may consume large memory).")

    train_ds = ds_cls(data=fold_train, transform=tr_tf, **ds_kwargs)
    val_ds = ds_cls(data=fold_val, transform=va_tf, **ds_kwargs)
    int_test_ds = ds_cls(data=int_test_files, transform=va_tf, **ds_kwargs)
    ext_test_ds = ds_cls(data=ext_test_files, transform=va_tf, **ds_kwargs)

    # Conservative workers helps reproducibility across platforms
    nw = max(0, min(2, int(args.num_workers)))

    g = torch.Generator()
    g.manual_seed(int(args.seed))

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, worker_init_fn=_seed_worker, generator=g, persistent_workers=(nw > 0)),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, worker_init_fn=_seed_worker, generator=g, persistent_workers=(nw > 0)),
        "internal_test": DataLoader(int_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, worker_init_fn=_seed_worker, generator=g, persistent_workers=(nw > 0)),
        "external": DataLoader(ext_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, worker_init_fn=_seed_worker, generator=g, persistent_workers=(nw > 0)),
    }
    return loaders


# =====================================================================
# Self-test (optional)
# =====================================================================
def _sanity_check(args: argparse.Namespace) -> None:
    log.info("--- Running Data Pipeline Sanity Check ---")
    loaders = get_fusion_dataloaders(args)
    batch = next(iter(loaders["train"]))

    log.info("Batch Integrity Check:")
    log.info(f"  image: {tuple(batch['image'].shape)} (expected: B,1,D,H,W)")
    log.info(f"  rad_features: {tuple(batch['rad_features'].shape)} (expected: B,N_feats)")
    log.info(f"  labels: {batch['label'].detach().cpu().tolist()}")

    if args.debug_print_batch and "id" in batch:
        log.info(f"  ids: {batch['id']}")


if __name__ == "__main__":
    try:
        a = get_args()
        a.use_cache = False
        a.batch_size = min(int(a.batch_size), 2)
        _sanity_check(a)
        log.info("--- Sanity Check PASSED ---")
    except Exception as e:
        log.error(f"Sanity Check FAILED: {e}", exc_info=True)
        sys.exit(1)
