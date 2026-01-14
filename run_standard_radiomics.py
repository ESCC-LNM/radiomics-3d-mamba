"""
PyRadiomics Feature Extraction Pipeline (3D NIfTI ROI)

What it does:
- Extract radiomics features from (image, mask) NIfTI pairs using PyRadiomics.
- Produces three CSV files (by default):
    radiomics_internal_trainval.csv
    radiomics_internal_test.csv
    radiomics_external_test.csv

Release / leakage notes:
- This script ONLY extracts features. It does NOT perform feature selection.
- By default it DOES NOT write labels into the output CSVs (safer; avoids leakage).
  Use --include_label only if you explicitly need labels inside the radiomics CSV.
- To avoid breaking downstream matching, the output CSV keeps a raw ID column ("ID") by default.
  If you want anonymized IDs for sharing, use --id_mode hash to add an "anon_id" column
  (and optionally --hash_id_in_csv if you really want to replace ID itself).

Typical downstream:
1) Feature selection (select_radiomics_features.py)
2) Training (train_fusion_mamba.py)

Example:
python run_standard_radiomics.py \
  --data_dir ./data \
  --output_dir ./outputs/radiomics_features

If you really want labels inside the CSV (NOT recommended for release):
  add: --include_label
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import radiomics
from radiomics import featureextractor
from tqdm import tqdm


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("radiomics_extractor")


log = setup_logger("INFO")


# ---------------------------------------------------------------------
# ID utils
# ---------------------------------------------------------------------
def clean_id(raw_id: Any) -> str:
    s = str(raw_id).strip()
    for suf in (".nii.gz", ".nii(1).gz", ".nii"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def stable_hash_id(raw: str, salt: str = "") -> str:
    """Stable anonymization (BLAKE2b)."""
    h = hashlib.blake2b(digest_size=10)
    msg = (salt + str(raw)).encode("utf-8", errors="ignore")
    h.update(msg)
    return f"case_{h.hexdigest()}"


def fmt_id_for_log(raw_id: str, id_mode: str, salt: str) -> str:
    """Avoid exposing raw IDs in logs unless user explicitly requests."""
    if id_mode == "hash":
        return stable_hash_id(raw_id, salt=salt)
    return str(raw_id)


# ---------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------
def require_exists(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")


def find_file_by_id(folder: Path, file_id: str, exts: Sequence[str]) -> Optional[Path]:
    """Find NIfTI by trying allowed extensions with given ID stem."""
    if not folder.exists():
        return None
    for ext in exts:
        p = folder / f"{file_id}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------
# Radiomics extractor
# ---------------------------------------------------------------------
def build_extractor(param_file: Optional[Path] = None) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Build a PyRadiomics extractor.
    If param_file is None, uses a robust default configuration (Original + LoG + Wavelet).
    """
    radiomics.setVerbosity(logging.ERROR)  # suppress internal messages

    if param_file is not None:
        require_exists(param_file, "PyRadiomics parameter file")
        log.info(f"Building extractor from param file: {param_file}")
        return featureextractor.RadiomicsFeatureExtractor(str(param_file))

    log.info("Building extractor with DEFAULT settings (Original, LoG, Wavelet enabled).")
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "normalize": True,
        "normalizeScale": 100,
        "label": 1,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
    extractor.enableImageTypeByName("Wavelet")

    extractor.enableAllFeatures()

    # Ensure 3D only: shape2D is not meaningful for 3D volumes.
    try:
        extractor.disableFeatureClassByName("shape2D")
    except Exception:
        # Some versions may behave differently; safe to ignore.
        pass

    return extractor


# ---------------------------------------------------------------------
# Metadata loading / validation
# ---------------------------------------------------------------------
def read_metadata(csv_path: Path) -> pd.DataFrame:
    require_exists(csv_path, "Metadata CSV")
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, required: Sequence[str], csv_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {missing}. Existing: {list(df.columns)}")


# ---------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------
def _filter_by_group(
    df_meta: pd.DataFrame,
    group_col: Optional[str],
    group_value: Optional[str],
) -> pd.DataFrame:
    if group_col and group_value and group_col in df_meta.columns:
        return df_meta[df_meta[group_col].astype(str).str.strip() == str(group_value)].copy()
    return df_meta.copy()


def extract_one_dataset(
    *,
    dataset_name: str,
    df_meta: pd.DataFrame,
    id_col: str,
    label_col: str,
    image_dir: Path,
    mask_dir: Path,
    output_csv: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    nifti_exts: Sequence[str],
    group_col: Optional[str],
    group_value: Optional[str],
    id_mode: str,
    id_salt: str,
    log_ids: bool,
    skip_existing: bool,
    include_label: bool,
    overwrite: bool,
    hash_id_in_csv: bool,
) -> None:
    if output_csv.exists() and skip_existing and not overwrite:
        log.info(f"[{dataset_name}] Output exists -> skip: {output_csv.name}")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_use = _filter_by_group(df_meta, group_col=group_col, group_value=group_value)
    if df_use.empty:
        log.warning(f"[{dataset_name}] No samples after filtering (group='{group_value}'). Skipping.")
        return

    # ID is always required. label is only required when include_label=True.
    validate_columns(df_use, [id_col], f"{dataset_name} metadata")
    if include_label:
        validate_columns(df_use, [label_col], f"{dataset_name} metadata")

    results: List[Dict[str, Any]] = []
    n_missing_img = 0
    n_missing_mask = 0
    n_failed = 0

    pbar = tqdm(df_use.itertuples(index=False), total=len(df_use), desc=f"[{dataset_name}] Extracting")
    for row in pbar:
        raw_id_val = getattr(row, id_col)
        file_id = clean_id(raw_id_val)

        img_path = find_file_by_id(image_dir, file_id, nifti_exts)
        if img_path is None:
            n_missing_img += 1
            if log_ids:
                log.warning(f"[{dataset_name}] Image missing: {fmt_id_for_log(file_id, id_mode, id_salt)}")
            continue

        msk_path = find_file_by_id(mask_dir, file_id, nifti_exts)
        if msk_path is None:
            n_missing_mask += 1
            if log_ids:
                log.warning(f"[{dataset_name}] Mask missing: {fmt_id_for_log(file_id, id_mode, id_salt)}")
            continue

        try:
            feats = extractor.execute(str(img_path), str(msk_path))
            feats = {k: v for k, v in feats.items() if not k.startswith("diagnostics_")}

            # Leakage control: label is OFF by default.
            if include_label:
                feats["label"] = int(getattr(row, label_col))

            # ID handling:
            # - Keep "ID" raw by default for downstream matching.
            # - If id_mode=hash, also include "anon_id" for privacy/logging.
            anon = stable_hash_id(file_id, salt=id_salt)
            if id_mode == "hash":
                feats["anon_id"] = anon

            if hash_id_in_csv:
                # If user explicitly wants to anonymize the ID column itself.
                feats["ID"] = anon
            else:
                feats["ID"] = file_id

            results.append(feats)

        except Exception as e:
            n_failed += 1
            if log_ids:
                log.error(f"[{dataset_name}] Failed on {fmt_id_for_log(file_id, id_mode, id_salt)}: {e}")
            else:
                log.error(f"[{dataset_name}] Failed on a case: {e}")

    if not results:
        log.warning(f"[{dataset_name}] No features extracted. Not writing CSV.")
        return

    df_out = pd.DataFrame(results)

    # Ensure ID is the first column
    if "ID" in df_out.columns:
        cols = ["ID"] + [c for c in df_out.columns if c != "ID"]
        df_out = df_out[cols]

    df_out.to_csv(output_csv, index=False)
    log.info(
        f"[{dataset_name}] Saved {output_csv.name} | "
        f"n={len(df_out)}, missing_img={n_missing_img}, missing_mask={n_missing_mask}, failed={n_failed}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyRadiomics Feature Extraction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_dir", type=Path, default=Path("./data"), help="Root data folder")

    # Keep compatibility:
    # - README may use --output_dir
    # - older commands may use --out_dir
    parser.add_argument(
        "--output_dir",
        dest="out_dir",
        type=Path,
        default=Path("./outputs/radiomics_features"),
        help="Output folder",
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        type=Path,
        default=Path("./outputs/radiomics_features"),
        help="(Alias) Output folder",
    )

    parser.add_argument("--param_file", type=Path, default=None, help="Optional PyRadiomics YAML config")

    # Metadata CSVs
    parser.add_argument("--trainval_csv", type=Path, default=Path("./data/internal_group.csv"))
    parser.add_argument("--internal_test_csv", type=Path, default=Path("./data/internal_group.csv"))
    parser.add_argument("--external_csv", type=Path, default=Path("./data/external_label.csv"))

    # Column mappings
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--group_col", type=str, default="group")

    # Leakage control
    parser.add_argument(
        "--include_label",
        action="store_true",
        help="[WARNING] If set, writes the target label into the output CSV. Default is OFF to prevent leakage.",
    )

    # ID privacy
    parser.add_argument(
        "--id_mode",
        type=str,
        choices=["raw", "hash"],
        default="raw",
        help="raw: keep raw IDs in logs/CSV. hash: add anon_id column and hash IDs in logs.",
    )
    parser.add_argument("--id_salt", type=str, default="", help="Salt for hash mode")
    parser.add_argument(
        "--hash_id_in_csv",
        action="store_true",
        help="[WARNING] Replace the CSV 'ID' column with hashed IDs. This may BREAK downstream matching unless the whole pipeline uses hashed IDs.",
    )

    # Image Directories (Relative to data_dir)
    parser.add_argument("--trainval_img_dir", type=str, default="trainval_images")
    parser.add_argument("--trainval_mask_dir", type=str, default="trainval_masks")
    parser.add_argument("--internal_test_img_dir", type=str, default="internal_test_images")
    parser.add_argument("--internal_test_mask_dir", type=str, default="internal_test_masks")
    parser.add_argument("--external_img_dir", type=str, default="external_test_images")
    parser.add_argument("--external_mask_dir", type=str, default="external_test_masks")

    # Group filtering
    parser.add_argument("--trainval_group_value", type=str, default="train")
    parser.add_argument("--internal_test_group_value", type=str, default="test")
    parser.add_argument("--external_group_value", type=str, default=None)

    # Output filenames
    parser.add_argument("--trainval_out", type=str, default="radiomics_internal_trainval.csv")
    parser.add_argument("--internal_test_out", type=str, default="radiomics_internal_test.csv")
    parser.add_argument("--external_out", type=str, default="radiomics_external_test.csv")

    # Misc
    parser.add_argument("--nifti_exts", type=str, nargs="+", default=[".nii.gz", ".nii", ".nii(1).gz"])
    parser.add_argument("--log_ids", action="store_true", help="Print IDs in logs for missing files (default off)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing if output CSV exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it exists")
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = get_args()
    global log
    log = setup_logger(args.log_level)

    log.info("--- Starting PyRadiomics Extraction Pipeline ---")
    log.info(f"Data Root: {args.data_dir}")
    log.info(f"Output Dir: {args.out_dir}")
    log.info(f"Include Labels in CSV: {args.include_label}")
    log.info(f"ID Mode: {args.id_mode} | hash_id_in_csv={args.hash_id_in_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_exists(args.data_dir, "data_dir")

    extractor = build_extractor(args.param_file)

    # Load metadata frames (skip gracefully if missing)
    def _safe_load(name: str, p: Path) -> pd.DataFrame:
        try:
            return read_metadata(p)
        except FileNotFoundError:
            log.warning(f"{name} CSV not found: {p}. Skipping {name}.")
            return pd.DataFrame()

    df_trainval = _safe_load("TrainVal", args.trainval_csv)
    df_internal_test = _safe_load("InternalTest", args.internal_test_csv)
    df_external = _safe_load("External", args.external_csv)

    datasets = [
        dict(
            name="internal_trainval",
            df=df_trainval,
            img_dir=args.data_dir / args.trainval_img_dir,
            mask_dir=args.data_dir / args.trainval_mask_dir,
            out_csv=args.out_dir / args.trainval_out,
            group_value=args.trainval_group_value,
        ),
        dict(
            name="internal_test",
            df=df_internal_test,
            img_dir=args.data_dir / args.internal_test_img_dir,
            mask_dir=args.data_dir / args.internal_test_mask_dir,
            out_csv=args.out_dir / args.internal_test_out,
            group_value=args.internal_test_group_value,
        ),
        dict(
            name="external_test",
            df=df_external,
            img_dir=args.data_dir / args.external_img_dir,
            mask_dir=args.data_dir / args.external_mask_dir,
            out_csv=args.out_dir / args.external_out,
            group_value=args.external_group_value,
        ),
    ]

    for cfg in datasets:
        if cfg["df"].empty:
            continue

        if not cfg["img_dir"].exists():
            log.warning(f"[{cfg['name']}] Image dir not found: {cfg['img_dir']}. Skipping.")
            continue
        if not cfg["mask_dir"].exists():
            log.warning(f"[{cfg['name']}] Mask dir not found: {cfg['mask_dir']}. Skipping.")
            continue

        extract_one_dataset(
            dataset_name=cfg["name"],
            df_meta=cfg["df"],
            id_col=args.id_col,
            label_col=args.label_col,
            image_dir=cfg["img_dir"],
            mask_dir=cfg["mask_dir"],
            output_csv=cfg["out_csv"],
            extractor=extractor,
            nifti_exts=args.nifti_exts,
            group_col=args.group_col,
            group_value=cfg["group_value"],
            id_mode=args.id_mode,
            id_salt=args.id_salt,
            log_ids=args.log_ids,
            skip_existing=args.skip_existing,
            include_label=args.include_label,
            overwrite=args.overwrite,
            hash_id_in_csv=args.hash_id_in_csv,
        )

    log.info("--- Feature Extraction Complete ---")


if __name__ == "__main__":
    main()