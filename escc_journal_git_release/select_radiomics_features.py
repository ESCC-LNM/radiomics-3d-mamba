"""
Leakage-controlled radiomics feature selection.

Inputs:
  - frozen master cohort manifest
  - frozen cross-validation manifest
  - raw radiomics tables for development, internal held-out, and external held-out cohorts

Outputs:
  - fold-specific radiomics tables for model selection (train/val only)
  - final radiomics tables for full-development training and held-out evaluation
  - selector metadata for each fold and for the final selector
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from experiment_utils import (
    configure_logging,
    ensure_dir,
    get_required,
    load_json,
    normalize_sample_id,
    require_columns,
    save_feature_list,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform leakage-controlled radiomics feature selection for both model selection and final training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional override for the selected-feature output root defined in the configuration file.",
    )
    return parser.parse_args()


def _read_radiomics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "ID"})
    df["ID"] = df["ID"].astype(str).map(normalize_sample_id)
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].tolist()[:10]
        raise ValueError(f"Duplicate radiomics sample identifiers were detected in {path.name}. Examples: {dup}")
    df = df.set_index("ID")
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        raise ValueError(f"No numeric radiomics features were found in {path.name}")
    return df


def _read_master_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
    require_columns(df, ["sample_id", "patient_id", "label", "cohort_role"], f"Manifest '{path.name}'")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).map(normalize_sample_id)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def _read_cv_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
    require_columns(df, ["sample_id", "patient_id", "label", "outer_fold", "fold_role", "cohort_role"], f"Manifest '{path.name}'")
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).map(normalize_sample_id)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["outer_fold"] = df["outer_fold"].astype(int)
    return df


def _auc_score(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return float("nan")
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return float("nan")
    return float(max(auc, 1.0 - auc))


def _correlation_prune(ranked_features: Sequence[str], X: pd.DataFrame, corr_threshold: float) -> List[str]:
    selected: List[str] = []
    for feat in ranked_features:
        keep = True
        for kept in selected:
            corr = np.corrcoef(X[feat].to_numpy(), X[kept].to_numpy())[0, 1]
            if np.isnan(corr):
                continue
            if abs(float(corr)) > float(corr_threshold):
                keep = False
                break
        if keep:
            selected.append(feat)
    return selected


def _select_features(X_train: pd.DataFrame, y_train: np.ndarray, top_k: int, min_auc: float, corr_threshold: float) -> Tuple[List[str], Dict[str, float]]:
    X_train = X_train.select_dtypes(include=[np.number]).copy()
    if X_train.empty:
        raise RuntimeError("No numeric radiomics features are available for supervised feature selection.")

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    variances = X_imp.var(axis=0)
    keep_cols = variances[variances > 0].index.tolist()
    X_imp = X_imp[keep_cols]
    if X_imp.empty:
        raise RuntimeError("All radiomics features were removed because they had zero variance on the fold-train subset.")

    auc_per_feature: Dict[str, float] = {}
    for col in X_imp.columns:
        auc = _auc_score(X_imp[col].to_numpy(), y_train)
        if not np.isnan(auc):
            auc_per_feature[col] = float(auc)

    if not auc_per_feature:
        raise RuntimeError("No radiomics feature survived univariate AUC screening on the fold-train subset.")

    ranked = sorted(auc_per_feature.keys(), key=lambda c: auc_per_feature[c], reverse=True)
    above_floor = [feat for feat in ranked if auc_per_feature[feat] >= float(min_auc)]
    shortlist = above_floor if len(above_floor) >= int(top_k) else ranked[: int(top_k)]
    selected = _correlation_prune(shortlist, X_imp[shortlist], corr_threshold=float(corr_threshold))
    if not selected:
        raise RuntimeError("Correlation pruning removed all candidate radiomics features.")
    return selected, auc_per_feature


def _align_feature_table(raw_table: pd.DataFrame, manifest_subset: pd.DataFrame) -> pd.DataFrame:
    sample_ids = manifest_subset["sample_id"].astype(str).tolist()
    missing = [sid for sid in sample_ids if sid not in raw_table.index]
    if missing:
        raise RuntimeError(
            "At least one manifest sample is absent from the radiomics table. "
            f"Examples: {missing[:10]}"
        )
    return raw_table.loc[sample_ids].copy()


def _build_selected_table(raw_table: pd.DataFrame, selected_features: Sequence[str]) -> pd.DataFrame:
    missing = [feat for feat in selected_features if feat not in raw_table.columns]
    if missing:
        raise RuntimeError(
            "A selected feature is absent from the downstream radiomics table. "
            f"Examples: {missing[:10]}"
        )
    out = raw_table.loc[:, list(selected_features)].copy()
    out.insert(0, "ID", out.index.astype(str))
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    output_root = args.output_root or Path(get_required(cfg, "paths.selected_features_root"))
    ensure_dir(output_root)
    logger = configure_logging(output_root / "select_radiomics_features.log")

    master_manifest = _read_master_manifest(Path(get_required(cfg, "paths.preprocessed_root")) / "master_cohort_manifest.csv")
    cv_manifest = _read_cv_manifest(Path(get_required(cfg, "paths.preprocessed_root")) / "cv_split_manifest.csv")

    development_raw = _read_radiomics(Path(get_required(cfg, "paths.radiomics.development_raw_csv")))
    internal_test_raw = _read_radiomics(Path(get_required(cfg, "paths.radiomics.internal_test_raw_csv")))
    external_test_raw = _read_radiomics(Path(get_required(cfg, "paths.radiomics.external_test_raw_csv")))

    top_k = int(get_required(cfg, "feature_selection.top_k"))
    min_auc = float(get_required(cfg, "feature_selection.min_auc"))
    corr_threshold = float(get_required(cfg, "feature_selection.corr_threshold"))
    n_splits = int(get_required(cfg, "cross_validation.n_splits"))

    final_dir = ensure_dir(output_root / "final")
    folds_dir = ensure_dir(output_root / "folds")

    development_master = master_manifest.loc[master_manifest["cohort_role"] == "development"].copy()
    internal_test_master = master_manifest.loc[master_manifest["cohort_role"] == "internal_test"].copy()
    external_role_name = str(get_required(cfg, "dataset_roles.external_test_role_name"))
    external_test_master = master_manifest.loc[master_manifest["cohort_role"] == external_role_name].copy()

    # Cross-validation feature selection: train only, applied to train/val within the same outer fold.
    fold_selector_rows: List[Dict[str, Any]] = []
    for fold_idx in range(1, n_splits + 1):
        fold_dir = ensure_dir(folds_dir / f"fold_{fold_idx:02d}")
        fold_rows = cv_manifest.loc[cv_manifest["outer_fold"] == int(fold_idx)].copy()
        train_rows = fold_rows.loc[fold_rows["fold_role"] == "train"].copy()
        val_rows = fold_rows.loc[fold_rows["fold_role"] == "val"].copy()

        X_train_raw = _align_feature_table(development_raw, train_rows)
        y_train = train_rows["label"].to_numpy(dtype=int)
        selected_features, auc_per_feature = _select_features(
            X_train_raw,
            y_train,
            top_k=top_k,
            min_auc=min_auc,
            corr_threshold=corr_threshold,
        )

        X_train_selected = _build_selected_table(X_train_raw, selected_features)
        X_val_selected = _build_selected_table(_align_feature_table(development_raw, val_rows), selected_features)
        X_train_selected.to_csv(fold_dir / "train_selected.csv", index=False)
        X_val_selected.to_csv(fold_dir / "val_selected.csv", index=False)

        save_feature_list(selected_features, fold_dir / "selected_features.txt")
        metadata = {
            "outer_fold": int(fold_idx),
            "n_train_samples": int(train_rows.shape[0]),
            "n_val_samples": int(val_rows.shape[0]),
            "n_selected_features": int(len(selected_features)),
            "selection_parameters": {
                "top_k": int(top_k),
                "min_auc": float(min_auc),
                "corr_threshold": float(corr_threshold),
            },
            "selected_features": list(selected_features),
            "auc_per_feature": auc_per_feature,
        }
        save_json(metadata, fold_dir / "selector_metadata.json")
        fold_selector_rows.append({
            "outer_fold": int(fold_idx),
            "n_selected_features": int(len(selected_features)),
        })
        logger.info("Completed supervised radiomics selection for outer fold %d.", fold_idx)

    # Final selector: fit on all development samples only, then apply to held-out cohorts.
    X_dev_raw = _align_feature_table(development_raw, development_master)
    y_dev = development_master["label"].to_numpy(dtype=int)
    final_selected_features, final_auc_per_feature = _select_features(
        X_dev_raw,
        y_dev,
        top_k=top_k,
        min_auc=min_auc,
        corr_threshold=corr_threshold,
    )

    final_development_selected = _build_selected_table(X_dev_raw, final_selected_features)
    final_internal_selected = _build_selected_table(_align_feature_table(internal_test_raw, internal_test_master), final_selected_features)
    final_external_selected = _build_selected_table(_align_feature_table(external_test_raw, external_test_master), final_selected_features)

    final_development_selected.to_csv(final_dir / "development_selected.csv", index=False)
    final_internal_selected.to_csv(final_dir / "internal_test_selected.csv", index=False)
    final_external_selected.to_csv(final_dir / "external_test_selected.csv", index=False)
    save_feature_list(final_selected_features, final_dir / "selected_features.txt")
    save_json(
        {
            "n_selected_features": int(len(final_selected_features)),
            "selection_parameters": {
                "top_k": int(top_k),
                "min_auc": float(min_auc),
                "corr_threshold": float(corr_threshold),
            },
            "selected_features": list(final_selected_features),
            "auc_per_feature": final_auc_per_feature,
        },
        final_dir / "selector_metadata.json",
    )
    save_json(
        {
            "folds": fold_selector_rows,
            "final_n_selected_features": int(len(final_selected_features)),
            "feature_selection_parameters": {
                "top_k": int(top_k),
                "min_auc": float(min_auc),
                "corr_threshold": float(corr_threshold),
            },
        },
        output_root / "feature_selection_summary.json",
    )

    logger.info("Cross-validation and final radiomics feature-selection artefacts were written to %s", output_root)


if __name__ == "__main__":
    main()
