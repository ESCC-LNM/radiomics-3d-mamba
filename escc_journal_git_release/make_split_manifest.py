"""
Create a frozen cohort manifest and a patient-level cross-validation manifest.

This script must be run before supervised radiomics feature selection or model
training. It is the single authoritative source of cohort roles and outer-fold
assignments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiment_utils import (
    configure_logging,
    ensure_dir,
    get_required,
    load_json,
    normalize_patient_id,
    normalize_sample_id,
    require_columns,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze cohort roles and patient-level outer folds for the multimodal fusion study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional override for the preprocessing output root defined in the configuration file.",
    )
    return parser.parse_args()


def _read_internal_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))
    metadata_path = Path(get_required(cfg, "paths.internal_metadata_csv"))

    df = pd.read_csv(metadata_path, dtype={sample_id_col: str, patient_id_col: str})
    require_columns(df, [sample_id_col, label_col, group_col, patient_id_col], f"Metadata file '{metadata_path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    df[patient_id_col] = df[patient_id_col].astype(str).map(normalize_patient_id)
    df[label_col] = df[label_col].astype(int)
    return df


def _read_external_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    patient_id_col = str(get_required(cfg, "columns.external_patient_id")) if "external_patient_id" in get_required(cfg, "columns") else None
    metadata_path = Path(get_required(cfg, "paths.external_metadata_csv"))

    dtypes = {sample_id_col: str}
    if patient_id_col is not None:
        dtypes[patient_id_col] = str
    df = pd.read_csv(metadata_path, dtype=dtypes)
    required = [sample_id_col, label_col]
    if patient_id_col is not None:
        required.append(patient_id_col)
    require_columns(df, required, f"Metadata file '{metadata_path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    df[label_col] = df[label_col].astype(int)
    if patient_id_col is not None:
        df[patient_id_col] = df[patient_id_col].astype(str).map(normalize_patient_id)
    else:
        df["external_patient_id_proxy"] = df[sample_id_col].astype(str)
        patient_id_col = "external_patient_id_proxy"
    df = df.rename(columns={patient_id_col: "_external_patient_id"})
    return df


def _validate_patient_consistency(df: pd.DataFrame, patient_id_col: str, label_col: str, role_name: str) -> None:
    n_unique = df.groupby(patient_id_col)[label_col].nunique(dropna=False)
    inconsistent = n_unique[n_unique > 1]
    if not inconsistent.empty:
        examples = inconsistent.index.astype(str).tolist()[:10]
        raise ValueError(
            f"At least one patient in the {role_name} cohort maps to more than one label. Examples: {examples}"
        )


def _build_outer_folds(df: pd.DataFrame, patient_id_col: str, label_col: str, n_splits: int, seed: int) -> Dict[str, int]:
    patient_df = df[[patient_id_col, label_col]].drop_duplicates(subset=[patient_id_col]).reset_index(drop=True)
    if patient_df.shape[0] < int(n_splits):
        raise ValueError(
            f"The number of unique development patients ({patient_df.shape[0]}) is smaller than n_splits ({n_splits})."
        )
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    assignments: Dict[str, int] = {}
    patient_ids = patient_df[patient_id_col].astype(str).to_numpy()
    labels = patient_df[label_col].astype(int).to_numpy()
    for fold_idx, (_, val_idx) in enumerate(splitter.split(np.zeros(len(patient_ids)), labels), start=1):
        for idx in val_idx:
            assignments[str(patient_ids[idx])] = int(fold_idx)
    return assignments


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    output_root = args.output_root or Path(get_required(cfg, "paths.preprocessed_root"))
    ensure_dir(output_root)
    logger = configure_logging(output_root / "make_split_manifest.log")

    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))

    development_group_value = str(get_required(cfg, "dataset_roles.development_group_value"))
    internal_test_group_value = str(get_required(cfg, "dataset_roles.internal_test_group_value"))
    external_test_role_name = str(get_required(cfg, "dataset_roles.external_test_role_name"))

    internal_metadata = _read_internal_metadata(cfg)
    external_metadata = _read_external_metadata(cfg)

    if internal_metadata[sample_id_col].duplicated().any():
        dup = internal_metadata.loc[internal_metadata[sample_id_col].duplicated(), sample_id_col].tolist()[:10]
        raise ValueError(f"Duplicate internal sample identifiers were detected. Examples: {dup}")
    if external_metadata[sample_id_col].duplicated().any():
        dup = external_metadata.loc[external_metadata[sample_id_col].duplicated(), sample_id_col].tolist()[:10]
        raise ValueError(f"Duplicate external sample identifiers were detected. Examples: {dup}")

    development_df = internal_metadata.loc[internal_metadata[group_col].astype(str) == development_group_value].copy()
    internal_test_df = internal_metadata.loc[internal_metadata[group_col].astype(str) == internal_test_group_value].copy()
    if development_df.empty:
        raise ValueError("No development samples were found after filtering by dataset_roles.development_group_value.")
    if internal_test_df.empty:
        raise ValueError("No internal held-out samples were found after filtering by dataset_roles.internal_test_group_value.")

    _validate_patient_consistency(development_df, patient_id_col, label_col, role_name="development")
    _validate_patient_consistency(internal_test_df, patient_id_col, label_col, role_name="internal held-out")

    overlap = set(development_df[patient_id_col].astype(str)) & set(internal_test_df[patient_id_col].astype(str))
    if overlap:
        examples = sorted(list(overlap))[:10]
        raise ValueError(
            "Patient-level leakage detected: at least one patient appears in both the development and internal held-out cohorts. "
            f"Examples: {examples}"
        )

    n_splits = int(get_required(cfg, "cross_validation.n_splits"))
    seed = int(get_required(cfg, "cross_validation.seed"))
    patient_to_fold = _build_outer_folds(development_df, patient_id_col, label_col, n_splits=n_splits, seed=seed)

    development_manifest = development_df[[sample_id_col, patient_id_col, label_col]].copy()
    development_manifest["cohort_role"] = "development"
    development_manifest["outer_fold"] = development_manifest[patient_id_col].map(patient_to_fold).astype(int)
    development_manifest["fold_role"] = "development"

    internal_manifest = internal_test_df[[sample_id_col, patient_id_col, label_col]].copy()
    internal_manifest["cohort_role"] = "internal_test"
    internal_manifest["outer_fold"] = pd.NA
    internal_manifest["fold_role"] = "heldout"

    external_manifest = external_metadata[[sample_id_col, "_external_patient_id", label_col]].copy()
    external_manifest = external_manifest.rename(columns={"_external_patient_id": patient_id_col})
    external_manifest["cohort_role"] = external_test_role_name
    external_manifest["outer_fold"] = pd.NA
    external_manifest["fold_role"] = "heldout"

    master_manifest = pd.concat([development_manifest, internal_manifest, external_manifest], axis=0, ignore_index=True)
    master_manifest = master_manifest.rename(
        columns={sample_id_col: "sample_id", patient_id_col: "patient_id", label_col: "label"}
    )
    master_manifest = master_manifest[["sample_id", "patient_id", "label", "cohort_role", "outer_fold", "fold_role"]].copy()
    master_manifest["sample_id"] = master_manifest["sample_id"].astype(str).map(normalize_sample_id)
    master_manifest["patient_id"] = master_manifest["patient_id"].astype(str).map(normalize_patient_id)

    if master_manifest["sample_id"].duplicated().any():
        dup = master_manifest.loc[master_manifest["sample_id"].duplicated(), "sample_id"].tolist()[:10]
        raise ValueError(f"The frozen manifest would contain duplicate sample identifiers. Examples: {dup}")

    cv_manifest_rows = []
    development_base = development_manifest.rename(
        columns={sample_id_col: "sample_id", patient_id_col: "patient_id", label_col: "label"}
    ).copy()
    assigned_fold = development_base["outer_fold"].astype(int).copy()
    for fold_idx in range(1, n_splits + 1):
        fold_df = development_base.copy()
        fold_df["assigned_outer_fold"] = assigned_fold
        fold_df["outer_fold"] = int(fold_idx)
        fold_df["fold_role"] = np.where(fold_df["assigned_outer_fold"] == int(fold_idx), "val", "train")
        fold_df = fold_df.drop(columns=["assigned_outer_fold"])
        cv_manifest_rows.append(fold_df)
    cv_manifest = pd.concat(cv_manifest_rows, axis=0, ignore_index=True)
    cv_manifest = cv_manifest[["sample_id", "patient_id", "label", "cohort_role", "outer_fold", "fold_role"]].copy()

    master_path = output_root / "master_cohort_manifest.csv"
    cv_path = output_root / "cv_split_manifest.csv"
    master_manifest.to_csv(master_path, index=False)
    cv_manifest.to_csv(cv_path, index=False)

    save_json(
        {
            "n_development_samples": int((master_manifest["cohort_role"] == "development").sum()),
            "n_internal_test_samples": int((master_manifest["cohort_role"] == "internal_test").sum()),
            "n_external_test_samples": int((master_manifest["cohort_role"] == external_test_role_name).sum()),
            "n_development_patients": int(master_manifest.loc[master_manifest["cohort_role"] == "development", "patient_id"].nunique()),
            "n_internal_test_patients": int(master_manifest.loc[master_manifest["cohort_role"] == "internal_test", "patient_id"].nunique()),
            "n_external_test_patients": int(master_manifest.loc[master_manifest["cohort_role"] == external_test_role_name, "patient_id"].nunique()),
            "n_splits": int(n_splits),
            "seed": int(seed),
        },
        output_root / "split_manifest_summary.json",
    )

    logger.info("Frozen cohort manifest written to %s", master_path)
    logger.info("Cross-validation split manifest written to %s", cv_path)


if __name__ == "__main__":
    main()
