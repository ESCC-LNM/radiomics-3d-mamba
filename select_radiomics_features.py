"""
Radiomics Feature Selection (Release Version)

Goal:
- Select informative radiomics features using ONLY training data.
- Prevent leakage in cross-validation by supporting fold-specific selection.

Modes:
1) global (default):
   - Use all internal samples with group==train_group_value to select features.
   - WARNING: If you later report K-Fold CV metrics on the same pool,
     global selection introduces feature-selection leakage into val folds.

2) fold:
   - Use StratifiedKFold on internal train pool, and select features ONLY on fold-train subset.
   - Outputs fold-specific selected CSVs and feature lists.
   - This is the recommended mode if you report K-Fold CV validation metrics.

Release-friendly:
- No private absolute paths by default (relative ./data).
- No ID logging by default.
- Explicit consistency checks (no silent dimension changes unless allow_missing_features=True).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("radiomics_feature_selection")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clean_id(x: object) -> str:
    s = str(x).strip()
    for suf in (".nii.gz", ".nii(1).gz", ".nii"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def read_metadata(
    path: Path,
    id_col: str,
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    df = pd.read_csv(path, dtype={id_col: str})
    for c in (id_col, label_col):
        if c not in df.columns:
            raise ValueError(f"Metadata CSV missing required column '{c}': {path}")

    if group_col not in df.columns:
        raise ValueError(
            f"Metadata CSV missing required column '{group_col}': {path}\n"
            f"This column is required to isolate train/test and prevent leakage."
        )

    df[id_col] = df[id_col].astype(str).map(clean_id)
    return df


def read_radiomics(path: Path) -> pd.DataFrame:
    """
    Radiomics CSV must contain an ID column (named 'ID' by default export).
    If not, we try to treat the first column as ID.
    """
    if not path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {path}")

    df = pd.read_csv(path)
    if "ID" not in df.columns:
        first_col = df.columns[0]
        log.warning(f"[{path.name}] No 'ID' column found. Using first column '{first_col}' as ID.")
        df = df.rename(columns={first_col: "ID"})

    df["ID"] = df["ID"].astype(str).map(clean_id)

    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].astype(str).unique().tolist()
        raise ValueError(f"[{path.name}] Duplicate IDs detected (showing up to 10): {dup[:10]}")

    df = df.set_index("ID")

    # Keep numeric columns; non-numeric coerced to NaN (handled by imputer)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        raise ValueError(f"[{path.name}] No numeric feature columns found.")
    return df


def compute_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Univariate AUC for a single feature.
    Direction does not matter: use max(auc, 1-auc).
    """
    # x should already be imputed (no NaN). Keep robust anyway:
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    if len(np.unique(y)) < 2:
        return np.nan
    if len(np.unique(x)) < 2:
        return np.nan

    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return np.nan

    return float(max(auc, 1.0 - auc))


def greedy_corr_filter(features_sorted: List[str], X: pd.DataFrame, corr_thresh: float) -> List[str]:
    """
    Greedy correlation pruning:
    - features_sorted is in descending importance order
    - keep a feature if its abs corr with ALL kept features <= corr_thresh
    """
    kept: List[str] = []
    for f in features_sorted:
        keep_flag = True
        for g in kept:
            r = np.corrcoef(X[f].values, X[g].values)[0, 1]
            if np.isnan(r):
                continue
            if abs(r) > corr_thresh:
                keep_flag = False
                break
        if keep_flag:
            kept.append(f)
    return kept


def select_features_from_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    top_k: int,
    min_auc: float,
    corr_thresh: float,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Feature selection using ONLY training data:
    - numeric columns
    - impute median (fit on train)
    - drop zero-variance
    - univariate AUC ranking
    - AUC threshold + top_k fallback
    - greedy correlation pruning
    """
    # Keep numeric columns only
    X_train = X_train.select_dtypes(include=[np.number]).copy()
    if X_train.shape[1] == 0:
        raise RuntimeError("No numeric radiomics features available after dtype filtering.")

    # Median impute (fit only on train subset)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )

    # Drop zero-variance (after impute)
    var = X_imp.var(axis=0)
    keep_cols = var[var > 0].index.tolist()
    X_imp = X_imp[keep_cols]
    if X_imp.shape[1] == 0:
        raise RuntimeError("All features have zero variance after imputation.")

    # Univariate AUC
    feat_auc: Dict[str, float] = {}
    for f in X_imp.columns:
        auc = compute_feature_auc(X_imp[f].values, y_train)
        if not np.isnan(auc):
            feat_auc[f] = float(auc)

    if not feat_auc:
        raise RuntimeError("All features have invalid AUC. Check labels/features.")

    # Sort by AUC desc
    sorted_feats = sorted(feat_auc.keys(), key=lambda k: feat_auc[k], reverse=True)

    # Apply min_auc; fallback to top_k
    strong_feats = [f for f in sorted_feats if feat_auc[f] >= min_auc]
    if len(strong_feats) < top_k:
        strong_feats = sorted_feats[:top_k]

    # Correlation pruning on the selected subset
    X_sub = X_imp[strong_feats]
    selected = greedy_corr_filter(strong_feats, X_sub, corr_thresh=corr_thresh)

    return selected, feat_auc


def ensure_features_exist(
    df: pd.DataFrame,
    selected: List[str],
    *,
    name: str,
    allow_missing: bool,
) -> List[str]:
    missing = [c for c in selected if c not in df.columns]
    if missing:
        msg = f"{name}: {len(missing)} selected features missing (showing up to 10): {missing[:10]}"
        if allow_missing:
            log.warning(msg + " -> will drop missing features.")
        else:
            raise ValueError(msg + " -> refusing to proceed (set --allow_missing_features to override).")
    return [c for c in selected if c in df.columns]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_global(args: argparse.Namespace) -> None:
    """
    Global feature selection on the entire internal train pool (group==train_group_value).

    NOTE (truthful):
    - If you do K-Fold CV on this same pool and report fold validation metrics,
      this global selection causes feature-selection leakage into val folds.
    """
    internal_csv = args.internal_csv
    rad_dir = args.radiomics_dir

    rad_train_csv = rad_dir / args.rad_train_name
    rad_int_test_csv = rad_dir / args.rad_int_test_name
    rad_ext_test_csv = rad_dir / args.rad_ext_test_name

    df_internal = read_metadata(internal_csv, args.id_col, args.label_col, args.group_col)

    df_rad_train = read_radiomics(rad_train_csv)
    df_rad_int_test = read_radiomics(rad_int_test_csv)
    df_rad_ext_test = read_radiomics(rad_ext_test_csv)

    # Train pool (group == train_group_value)
    df_train_meta = df_internal[df_internal[args.group_col].astype(str).str.strip() == args.train_group_value].copy()
    if df_train_meta.empty:
        raise RuntimeError(f"No samples found with {args.group_col} == '{args.train_group_value}' in internal CSV.")

    df_train_meta["ID_clean"] = df_train_meta[args.id_col].map(clean_id)
    df_train_meta = df_train_meta[["ID_clean", args.label_col]].dropna()

    # Align IDs with radiomics train table
    common_ids = sorted(set(df_train_meta["ID_clean"]).intersection(df_rad_train.index))
    if len(common_ids) == 0:
        raise RuntimeError("Train pool IDs do not match radiomics train CSV. Check ID normalization.")

    df_train_meta = df_train_meta.set_index("ID_clean").loc[common_ids]
    X_train = df_rad_train.loc[common_ids].copy()
    y_train = df_train_meta[args.label_col].values.astype(int)

    # Binary label check
    uniq = np.unique(y_train)
    if len(uniq) != 2:
        raise ValueError(f"Binary classification expected, but got labels: {uniq.tolist()}")

    log.info(f"[global] Effective train samples for FS: {X_train.shape[0]}")

    selected, feat_auc = select_features_from_train(
        X_train,
        y_train,
        top_k=args.top_k,
        min_auc=args.min_auc,
        corr_thresh=args.corr_thresh,
    )
    log.info(f"[global] Selected features after corr filter: {len(selected)}")

    # Enforce same features exist across all splits (no silent mismatch)
    selected = ensure_features_exist(df_rad_train, selected, name="Train radiomics", allow_missing=args.allow_missing_features)
    selected = ensure_features_exist(df_rad_int_test, selected, name="Internal test radiomics", allow_missing=args.allow_missing_features)
    selected = ensure_features_exist(df_rad_ext_test, selected, name="External test radiomics", allow_missing=args.allow_missing_features)

    if len(selected) == 0:
        raise RuntimeError("No common selected features across all splits after checks.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = df_rad_train[selected].copy().reset_index()
    out_int = df_rad_int_test[selected].copy().reset_index()
    out_ext = df_rad_ext_test[selected].copy().reset_index()

    sel_train_path = out_dir / "radiomics_internal_trainval_sel.csv"
    sel_int_path = out_dir / "radiomics_internal_test_sel.csv"
    sel_ext_path = out_dir / "radiomics_external_test_sel.csv"
    sel_list_path = out_dir / "selected_features.txt"

    out_train.to_csv(sel_train_path, index=False)
    out_int.to_csv(sel_int_path, index=False)
    out_ext.to_csv(sel_ext_path, index=False)

    with open(sel_list_path, "w", encoding="utf-8") as f:
        for feat in selected:
            f.write(f"{feat}\t{feat_auc.get(feat, np.nan):.6f}\n")

    log.info(f"[global] Saved: {sel_train_path}")
    log.info(f"[global] Saved: {sel_int_path}")
    log.info(f"[global] Saved: {sel_ext_path}")
    log.info(f"[global] Saved: {sel_list_path}")
    log.info("[global] Done. Use these *_sel.csv in your fusion pipeline.")


def run_fold(args: argparse.Namespace) -> None:
    """
    Fold-specific feature selection:
    - Split internal train pool into (fold_train, fold_val) using StratifiedKFold
    - Select features ONLY on fold_train
    - Output fold-specific selected CSVs and feature list

    This avoids feature-selection leakage into val fold when reporting CV metrics.
    """
    internal_csv = args.internal_csv
    rad_dir = args.radiomics_dir

    rad_train_csv = rad_dir / args.rad_train_name
    rad_int_test_csv = rad_dir / args.rad_int_test_name
    rad_ext_test_csv = rad_dir / args.rad_ext_test_name

    df_internal = read_metadata(internal_csv, args.id_col, args.label_col, args.group_col)

    df_rad_train = read_radiomics(rad_train_csv)
    df_rad_int_test = read_radiomics(rad_int_test_csv)
    df_rad_ext_test = read_radiomics(rad_ext_test_csv)

    df_pool = df_internal[df_internal[args.group_col].astype(str).str.strip() == args.train_group_value].copy()
    if df_pool.empty:
        raise RuntimeError(f"No samples found with {args.group_col} == '{args.train_group_value}' in internal CSV.")

    df_pool["ID_clean"] = df_pool[args.id_col].map(clean_id)
    df_pool = df_pool[["ID_clean", args.label_col]].dropna()

    # Align to radiomics train table
    common_ids = sorted(set(df_pool["ID_clean"]).intersection(df_rad_train.index))
    if len(common_ids) == 0:
        raise RuntimeError("Train pool IDs do not match radiomics train CSV. Check ID normalization.")

    df_pool = df_pool.set_index("ID_clean").loc[common_ids]
    X_pool = df_rad_train.loc[common_ids].copy()
    y_pool = df_pool[args.label_col].values.astype(int)

    uniq = np.unique(y_pool)
    if len(uniq) != 2:
        raise ValueError(f"Binary classification expected, but got labels: {uniq.tolist()}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(y_pool)), y_pool))

    if not (0 <= args.fold_idx < len(splits)):
        raise IndexError(f"fold_idx={args.fold_idx} out of range for n_splits={args.n_splits}")

    tr_idx, va_idx = splits[args.fold_idx]
    train_ids = [common_ids[i] for i in tr_idx]
    val_ids = [common_ids[i] for i in va_idx]

    X_train = df_rad_train.loc[train_ids].copy()
    y_train = df_pool.loc[train_ids, args.label_col].values.astype(int)

    log.info(f"[fold] Fold {args.fold_idx + 1}/{args.n_splits}")
    log.info(f"[fold] FS train subset: n={len(train_ids)}, val subset: n={len(val_ids)}")

    selected, feat_auc = select_features_from_train(
        X_train,
        y_train,
        top_k=args.top_k,
        min_auc=args.min_auc,
        corr_thresh=args.corr_thresh,
    )
    log.info(f"[fold] Selected features after corr filter: {len(selected)}")

    # Enforce same features exist across all splits for stable dimension
    selected = ensure_features_exist(df_rad_train, selected, name="Train radiomics", allow_missing=args.allow_missing_features)
    selected = ensure_features_exist(df_rad_int_test, selected, name="Internal test radiomics", allow_missing=args.allow_missing_features)
    selected = ensure_features_exist(df_rad_ext_test, selected, name="External test radiomics", allow_missing=args.allow_missing_features)

    if len(selected) == 0:
        raise RuntimeError("No common selected features across all splits after checks.")

    out_dir = args.out_dir / f"fold{args.fold_idx + 1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = df_rad_train[selected].copy().reset_index()
    out_int = df_rad_int_test[selected].copy().reset_index()
    out_ext = df_rad_ext_test[selected].copy().reset_index()

    sel_train_path = out_dir / "radiomics_internal_trainval_sel.csv"
    sel_int_path = out_dir / "radiomics_internal_test_sel.csv"
    sel_ext_path = out_dir / "radiomics_external_test_sel.csv"
    sel_list_path = out_dir / "selected_features.txt"

    out_train.to_csv(sel_train_path, index=False)
    out_int.to_csv(sel_int_path, index=False)
    out_ext.to_csv(sel_ext_path, index=False)

    with open(sel_list_path, "w", encoding="utf-8") as f:
        for feat in selected:
            f.write(f"{feat}\t{feat_auc.get(feat, np.nan):.6f}\n")

    log.info(f"[fold] Saved: {sel_train_path}")
    log.info(f"[fold] Saved: {sel_int_path}")
    log.info(f"[fold] Saved: {sel_ext_path}")
    log.info(f"[fold] Saved: {sel_list_path}")
    log.info("[fold] Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select informative radiomics features using ONLY internal training data (release-safe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # GitHub-friendly defaults
    parser.add_argument("--internal_csv", type=Path, default=Path("./data/internal_group.csv"))
    parser.add_argument("--radiomics_dir", type=Path, default=Path("./outputs/radiomics_features"))
    parser.add_argument("--out_dir", type=Path, default=None, help="Output folder. If omitted: global->./outputs/selected_features/global, fold->./outputs/selected_features")

    # Metadata schema
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--group_col", type=str, default="group")
    parser.add_argument("--train_group_value", type=str, default="train")

    # Radiomics file names inside radiomics_dir
    parser.add_argument("--rad_train_name", type=str, default="radiomics_internal_trainval.csv")
    parser.add_argument("--rad_int_test_name", type=str, default="radiomics_internal_test.csv")
    parser.add_argument("--rad_ext_test_name", type=str, default="radiomics_external_test.csv")

    # Selection params
    parser.add_argument("--top_k", type=int, default=112, help="Target max number of features after AUC ranking (paper default: 112).")
    parser.add_argument("--min_auc", type=float, default=0.60, help="Minimum univariate AUC to be considered strong.")
    parser.add_argument("--corr_thresh", type=float, default=0.90, help="Correlation threshold for redundancy pruning.")

    # Leakage control mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["global", "fold"],
        default="global",
        help="global: select using all internal train pool; fold: select using fold-train only (recommended for CV).",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Used when mode=fold.")
    parser.add_argument("--fold_idx", type=int, default=0, help="0-based fold index when mode=fold.")
    parser.add_argument("--seed", type=int, default=42, help="Used when mode=fold.")

    # Strictness
    parser.add_argument(
        "--allow_missing_features",
        action="store_true",
        help="If set, drop selected features missing in a split instead of failing. Not recommended for paper release.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Default output folder depends on mode (avoid fold/global mixing)
    if args.out_dir is None:
        args.out_dir = Path('./outputs/selected_features/global') if args.mode == 'global' else Path('./outputs/selected_features')

    # Basic existence checks
    if not args.internal_csv.exists():
        raise FileNotFoundError(f"internal_csv not found: {args.internal_csv}")
    if not args.radiomics_dir.exists():
        raise FileNotFoundError(f"radiomics_dir not found: {args.radiomics_dir}")

    log.info(f"Mode           : {args.mode}")
    log.info(f"Radiomics dir  : {args.radiomics_dir}")
    log.info(f"Output dir     : {args.out_dir}")

    if args.mode == "global":
        run_global(args)
    else:
        run_fold(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Feature selection FAILED: {e}", exc_info=True)
        raise
