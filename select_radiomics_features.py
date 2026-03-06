from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("radiomics_feature_selection_v2")


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def clean_id(x: object) -> str:
    s = str(x).strip()
    for suf in (".nii.gz", ".nii(1).gz", ".nii"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s


def read_metadata(path: Path, id_col: str, label_col: str, group_col: str, patient_id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path, dtype={id_col: str, patient_id_col: str})
    for c in (id_col, label_col, group_col, patient_id_col):
        if c not in df.columns:
            raise ValueError(f"Metadata CSV missing required column '{c}': {path}")
    df[id_col] = df[id_col].astype(str).map(clean_id)
    df[patient_id_col] = df[patient_id_col].astype(str).str.strip()
    return df


def read_radiomics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        first_col = df.columns[0]
        log.warning("[%s] No 'ID' column found. Using first column '%s' as ID.", path.name, first_col)
        df = df.rename(columns={first_col: "ID"})
    df["ID"] = df["ID"].astype(str).map(clean_id)
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].tolist()[:10]
        raise ValueError(f"[{path.name}] Duplicate IDs detected: {dup}")
    df = df.set_index("ID")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        raise ValueError(f"[{path.name}] No numeric radiomics features found.")
    return df


# -----------------------------------------------------------------------------
# Selection logic
# -----------------------------------------------------------------------------
def compute_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return np.nan
    return float(max(auc, 1.0 - auc))


def greedy_corr_filter(features_sorted: List[str], X: pd.DataFrame, corr_thresh: float, top_k: int) -> List[str]:
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
        if len(kept) >= int(top_k):
            break
    return kept


def select_features_from_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    top_k: int,
    min_auc: float,
    corr_thresh: float,
) -> Tuple[List[str], Dict[str, float]]:
    X_train = X_train.select_dtypes(include=[np.number]).copy()
    if X_train.shape[1] == 0:
        raise RuntimeError("No numeric radiomics features available after dtype filtering.")

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    var = X_imp.var(axis=0)
    keep_cols = var[var > 0].index.tolist()
    X_imp = X_imp[keep_cols]
    if X_imp.shape[1] == 0:
        raise RuntimeError("All features have zero variance after imputation.")

    feat_auc: Dict[str, float] = {}
    for f in X_imp.columns:
        auc = compute_feature_auc(X_imp[f].values, y_train)
        if not np.isnan(auc):
            feat_auc[f] = float(auc)
    if not feat_auc:
        raise RuntimeError("All features have invalid AUC. Check labels/features.")

    sorted_feats = sorted(feat_auc.keys(), key=lambda k: feat_auc[k], reverse=True)
    candidate_feats = [f for f in sorted_feats if feat_auc[f] >= float(min_auc)]
    if len(candidate_feats) < int(top_k):
        candidate_feats = sorted_feats

    selected = greedy_corr_filter(candidate_feats, X_imp[candidate_feats], float(corr_thresh), int(top_k))
    if not selected:
        raise RuntimeError("No features survived correlation filtering.")
    return selected, feat_auc


# -----------------------------------------------------------------------------
# Split manifest
# -----------------------------------------------------------------------------
def validate_patient_labels(df: pd.DataFrame, label_col: str, patient_id_col: str) -> None:
    patient_nuniq = df.groupby(patient_id_col)[label_col].nunique(dropna=True)
    bad = patient_nuniq[patient_nuniq > 1]
    if len(bad) > 0:
        example = bad.index.astype(str).tolist()[:10]
        raise ValueError(
            "Found patients with conflicting labels. Grouped CV requires one consistent label per patient. "
            f"Examples: {example}"
        )


def build_split_manifest(
    df_internal: pd.DataFrame,
    df_rad_train: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
    group_col: str,
    patient_id_col: str,
    train_group_value: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    pool = df_internal[df_internal[group_col].astype(str).str.strip() == str(train_group_value)].copy()
    if pool.empty:
        raise RuntimeError(f"No samples found with {group_col} == '{train_group_value}'.")

    pool = pool[[id_col, label_col, patient_id_col]].dropna().copy()
    pool[id_col] = pool[id_col].map(clean_id)
    pool[patient_id_col] = pool[patient_id_col].astype(str).str.strip()
    pool = pool[pool[id_col].isin(df_rad_train.index)].copy()

    if pool.empty:
        raise RuntimeError("No overlap between internal train metadata and radiomics train CSV.")
    if pool[id_col].duplicated().any():
        dup = pool.loc[pool[id_col].duplicated(), id_col].tolist()[:10]
        raise ValueError(f"Duplicate sample IDs in training metadata: {dup}")

    validate_patient_labels(pool, label_col=label_col, patient_id_col=patient_id_col)

    y = pool[label_col].astype(int).values
    groups = pool[patient_id_col].astype(str).values
    if len(np.unique(y)) != 2:
        raise ValueError(f"Binary labels expected, but got {sorted(np.unique(y).tolist())}")

    fold_assign = np.full(len(pool), -1, dtype=int)

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        split_iter = splitter.split(np.zeros(len(pool)), y, groups)
        split_name = "StratifiedGroupKFold"
    else:
        splitter = GroupKFold(n_splits=int(n_splits))
        split_iter = splitter.split(np.zeros(len(pool)), y, groups)
        split_name = "GroupKFold"
        log.warning("StratifiedGroupKFold unavailable; falling back to GroupKFold (still patient-safe, but less balanced).")

    for fold_idx, (_, val_idx) in enumerate(split_iter):
        fold_assign[val_idx] = int(fold_idx)

    if np.any(fold_assign < 0):
        raise RuntimeError("Some samples were not assigned to any fold.")

    out = pool.rename(columns={id_col: "ID", label_col: "label", patient_id_col: "patient_id"}).copy()
    out["fold"] = fold_assign
    out = out[["ID", "label", "patient_id", "fold"]].sort_values(["fold", "patient_id", "ID"]).reset_index(drop=True)
    log.info("Built patient-level CV manifest using %s with %d samples and %d unique patients.", split_name, len(out), out["patient_id"].nunique())
    return out


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------
def ensure_features_exist(df: pd.DataFrame, selected: List[str], name: str) -> List[str]:
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: {len(missing)} selected features missing. Examples: {missing[:10]}")
    return selected


def write_selected_tables(
    out_dir: Path,
    selected: List[str],
    feat_auc: Dict[str, float],
    df_rad_train: pd.DataFrame,
    df_rad_int_test: pd.DataFrame,
    df_rad_ext_test: pd.DataFrame,
) -> Dict[str, Any]:
    selected = ensure_features_exist(df_rad_train, selected, "Train radiomics")
    selected = ensure_features_exist(df_rad_int_test, selected, "Internal test radiomics")
    selected = ensure_features_exist(df_rad_ext_test, selected, "External test radiomics")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "radiomics_internal_trainval_sel.csv"
    int_path = out_dir / "radiomics_internal_test_sel.csv"
    ext_path = out_dir / "radiomics_external_test_sel.csv"
    feat_path = out_dir / "selected_features.txt"

    df_rad_train[selected].copy().reset_index().to_csv(train_path, index=False)
    df_rad_int_test[selected].copy().reset_index().to_csv(int_path, index=False)
    df_rad_ext_test[selected].copy().reset_index().to_csv(ext_path, index=False)

    with open(feat_path, "w", encoding="utf-8") as f:
        for feat in selected:
            f.write(f"{feat}\t{feat_auc.get(feat, np.nan):.6f}\n")

    return {
        "n_features": int(len(selected)),
        "train_csv": str(train_path),
        "internal_test_csv": str(int_path),
        "external_test_csv": str(ext_path),
        "feature_list_txt": str(feat_path),
    }


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    rad_train_csv = Path(args.radiomics_dir) / args.rad_train_name
    rad_int_test_csv = Path(args.radiomics_dir) / args.rad_int_test_name
    rad_ext_test_csv = Path(args.radiomics_dir) / args.rad_ext_test_name

    df_internal = read_metadata(
        Path(args.internal_csv),
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=args.group_col,
        patient_id_col=args.patient_id_col,
    )
    df_rad_train = read_radiomics(rad_train_csv)
    df_rad_int_test = read_radiomics(rad_int_test_csv)
    df_rad_ext_test = read_radiomics(rad_ext_test_csv)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = build_split_manifest(
        df_internal,
        df_rad_train,
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=args.group_col,
        patient_id_col=args.patient_id_col,
        train_group_value=args.train_group_value,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    manifest_path = out_root / "cv_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log.info("Saved split manifest: %s", manifest_path)

    summary: Dict[str, Any] = {
        "mode": args.mode,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "split_manifest_csv": str(manifest_path),
        "patient_id_col": str(args.patient_id_col),
        "group_col": str(args.group_col),
        "train_group_value": str(args.train_group_value),
        "top_k": int(args.top_k),
        "min_auc": float(args.min_auc),
        "corr_thresh": float(args.corr_thresh),
        "cv_folds": [],
    }

    if args.mode in {"cv", "all"}:
        for fold_idx in range(int(args.n_splits)):
            train_ids = manifest.loc[manifest["fold"] != fold_idx, "ID"].astype(str).tolist()
            val_ids = manifest.loc[manifest["fold"] == fold_idx, "ID"].astype(str).tolist()
            X_train = df_rad_train.loc[train_ids].copy()
            y_train = manifest.set_index("ID").loc[train_ids, "label"].astype(int).values

            selected, feat_auc = select_features_from_train(
                X_train,
                y_train,
                top_k=args.top_k,
                min_auc=args.min_auc,
                corr_thresh=args.corr_thresh,
            )
            fold_dir = out_root / f"fold{fold_idx + 1:02d}"
            fold_meta = write_selected_tables(
                fold_dir,
                selected,
                feat_auc,
                df_rad_train,
                df_rad_int_test,
                df_rad_ext_test,
            )
            fold_meta.update(
                {
                    "fold": int(fold_idx),
                    "train_samples": int(len(train_ids)),
                    "val_samples": int(len(val_ids)),
                }
            )
            summary["cv_folds"].append(fold_meta)
            log.info("[CV] Fold %02d done | train=%d | val=%d | features=%d", fold_idx + 1, len(train_ids), len(val_ids), len(selected))

    if args.mode in {"final", "all"}:
        global_ids = manifest["ID"].astype(str).tolist()
        X_global = df_rad_train.loc[global_ids].copy()
        y_global = manifest.set_index("ID").loc[global_ids, "label"].astype(int).values
        selected, feat_auc = select_features_from_train(
            X_global,
            y_global,
            top_k=args.top_k,
            min_auc=args.min_auc,
            corr_thresh=args.corr_thresh,
        )
        final_dir = out_root / "final"
        final_meta = write_selected_tables(
            final_dir,
            selected,
            feat_auc,
            df_rad_train,
            df_rad_int_test,
            df_rad_ext_test,
        )
        final_meta.update({"train_samples": int(len(global_ids))})
        summary["final"] = final_meta
        log.info("[FINAL] Global feature selection done | train=%d | features=%d", len(global_ids), len(selected))

    summary_path = out_root / "selection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Saved selection summary: %s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-safe radiomics feature selection for CV tuning + final training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--internal_csv", type=Path, default=Path("./data3D/internal_group.csv"))
    parser.add_argument("--radiomics_dir", type=Path, default=Path("./outputs/radiomics_features"))
    parser.add_argument("--out_dir", type=Path, default=Path("./outputs/selected_features"))

    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--group_col", type=str, default="group")
    parser.add_argument("--patient_id_col", type=str, default="patient_id")
    parser.add_argument("--train_group_value", type=str, default="train")

    parser.add_argument("--rad_train_name", type=str, default="radiomics_internal_trainval.csv")
    parser.add_argument("--rad_int_test_name", type=str, default="radiomics_internal_test.csv")
    parser.add_argument("--rad_ext_test_name", type=str, default="radiomics_external_test.csv")

    parser.add_argument("--top_k", type=int, default=112)
    parser.add_argument("--min_auc", type=float, default=0.60)
    parser.add_argument("--corr_thresh", type=float, default=0.90)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["cv", "final", "all"], default="all")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
