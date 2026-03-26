"""
Configuration-driven radiomics extraction script.

This script performs feature extraction only. It does not perform any
label-dependent filtering, normalisation, or supervised feature selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from experiment_utils import (
    configure_logging,
    ensure_dir,
    get_optional,
    get_required,
    load_json,
    normalize_patient_id,
    normalize_sample_id,
    require_columns,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract radiomics features for the development, internal held-out, and external held-out cohorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional override for the radiomics output root defined in the configuration file.",
    )
    return parser.parse_args()


def _read_internal_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))

    path = Path(get_required(cfg, "paths.internal_metadata_csv"))
    df = pd.read_csv(path, dtype={sample_id_col: str, patient_id_col: str})
    require_columns(df, [sample_id_col, label_col, group_col, patient_id_col], f"Metadata file '{path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    df[patient_id_col] = df[patient_id_col].astype(str).map(normalize_patient_id)
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
    required = [sample_id_col, label_col]
    if external_patient_id_col is not None:
        required.append(str(external_patient_id_col))
    require_columns(df, required, f"Metadata file '{path.name}'")
    df = df.copy()
    df[sample_id_col] = df[sample_id_col].astype(str).map(normalize_sample_id)
    if external_patient_id_col is not None:
        df[str(external_patient_id_col)] = df[str(external_patient_id_col)].astype(str).map(normalize_patient_id)
    return df


def _build_extractor(cfg: Mapping[str, Any]):
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "PyRadiomics is required for run_standard_radiomics.py. Install it with the project requirements."
        ) from exc

    parameters_yaml = get_optional(cfg, "radiomics_extraction.parameters_yaml", default=None)
    if parameters_yaml:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(parameters_yaml))
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    enabled_image_types = get_optional(cfg, "radiomics_extraction.enabled_image_types", default=None)
    if enabled_image_types:
        extractor.disableAllImageTypes()
        for image_type, kwargs in enabled_image_types.items():
            extractor.enableImageTypeByName(image_type, customArgs=kwargs or {})

    enabled_feature_classes = get_optional(cfg, "radiomics_extraction.enabled_feature_classes", default=None)
    if enabled_feature_classes:
        extractor.disableAllFeatures()
        for cls_name, feats in enabled_feature_classes.items():
            if feats:
                extractor.enableFeaturesByName(**{cls_name: feats})
            else:
                extractor.enableFeatureClassByName(cls_name)

    settings = get_optional(cfg, "radiomics_extraction.settings", default={}) or {}
    for key, value in settings.items():
        extractor.settings[key] = value
    return extractor


def _extract_subset(
    rows: pd.DataFrame,
    *,
    sample_id_col: str,
    image_dir: Path,
    mask_dir: Path,
    allowed_suffixes: List[str],
    extractor,
    logger,
    include_label: bool,
    include_patient_id: bool,
    label_col: str,
    patient_id_col: Optional[str],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        sample_id = normalize_sample_id(row[sample_id_col])
        image_path = resolve_image_path(image_dir, sample_id, allowed_suffixes)
        mask_path = resolve_image_path(mask_dir, sample_id, allowed_suffixes)
        feature_vector = extractor.execute(str(image_path), str(mask_path))
        clean_features = {
            str(k): v
            for k, v in feature_vector.items()
            if not str(k).startswith("diagnostics_")
        }
        clean_features["ID"] = sample_id
        if include_label:
            clean_features[label_col] = int(row[label_col])
        if include_patient_id and patient_id_col is not None:
            clean_features[patient_id_col] = str(row[patient_id_col])
        records.append(clean_features)
    out = pd.DataFrame(records)
    logger.info("Radiomics extracted for %d samples.", len(out))
    return out


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    output_root = args.output_root or Path(get_required(cfg, "paths.radiomics_output_root"))
    ensure_dir(output_root)
    logger = configure_logging(output_root / "run_standard_radiomics.log")

    sample_id_col = str(get_required(cfg, "columns.sample_id"))
    label_col = str(get_required(cfg, "columns.label"))
    group_col = str(get_required(cfg, "columns.group"))
    patient_id_col = str(get_required(cfg, "columns.patient_id"))

    development_group_value = str(get_required(cfg, "dataset_roles.development_group_value"))
    internal_test_group_value = str(get_required(cfg, "dataset_roles.internal_test_group_value"))

    allowed_suffixes = [str(v) for v in get_required(cfg, "dataset.allowed_image_suffixes")]
    include_label = bool(get_optional(cfg, "radiomics_extraction.include_label_column", default=False))
    include_patient_id = bool(get_optional(cfg, "radiomics_extraction.include_patient_id_column", default=False))

    internal_metadata = _read_internal_metadata(cfg)
    external_metadata = _read_external_metadata(cfg)

    development_rows = internal_metadata.loc[internal_metadata[group_col].astype(str) == development_group_value].copy()
    internal_test_rows = internal_metadata.loc[internal_metadata[group_col].astype(str) == internal_test_group_value].copy()
    external_rows = external_metadata.copy()

    extractor = _build_extractor(cfg)

    outputs = {
        "development": {
            "rows": development_rows,
            "image_dir": Path(get_required(cfg, "paths.images.development_dir")),
            "mask_dir": Path(get_required(cfg, "paths.masks.development_dir")),
            "output_path": Path(get_required(cfg, "paths.radiomics.development_raw_csv")),
            "patient_id_col": patient_id_col,
        },
        "internal_test": {
            "rows": internal_test_rows,
            "image_dir": Path(get_required(cfg, "paths.images.internal_test_dir")),
            "mask_dir": Path(get_required(cfg, "paths.masks.internal_test_dir")),
            "output_path": Path(get_required(cfg, "paths.radiomics.internal_test_raw_csv")),
            "patient_id_col": patient_id_col,
        },
        "external_test": {
            "rows": external_rows,
            "image_dir": Path(get_required(cfg, "paths.images.external_test_dir")),
            "mask_dir": Path(get_required(cfg, "paths.masks.external_test_dir")),
            "output_path": Path(get_required(cfg, "paths.radiomics.external_test_raw_csv")),
            "patient_id_col": get_optional(cfg, "columns.external_patient_id", default=None),
        },
    }

    for subset_name, bundle in outputs.items():
        if bundle["rows"].empty:
            raise ValueError(f"Subset '{subset_name}' is empty. Check the metadata and cohort-role configuration.")
        table = _extract_subset(
            bundle["rows"],
            sample_id_col=sample_id_col,
            image_dir=Path(bundle["image_dir"]),
            mask_dir=Path(bundle["mask_dir"]),
            allowed_suffixes=allowed_suffixes,
            extractor=extractor,
            logger=logger,
            include_label=include_label,
            include_patient_id=include_patient_id,
            label_col=label_col,
            patient_id_col=bundle["patient_id_col"],
        )
        output_path = Path(bundle["output_path"])
        ensure_dir(output_path.parent)
        table.to_csv(output_path, index=False)
        logger.info("Saved %s radiomics table to %s", subset_name, output_path)


if __name__ == "__main__":
    main()
