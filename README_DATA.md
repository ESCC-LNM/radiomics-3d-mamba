# Data Preparation Guide

This repository does not include any data. Prepare your own dataset under `./data/`.

Privacy rules for public release:

- Do not upload patient identifiers.
- Do not upload images, masks, checkpoints, logs, or radiomics tables.
- Do not paste private filesystem paths into issues or pull requests.

## Required Directory Structure

```text
data/
|-- trainval_images/
|-- trainval_masks/
|-- internal_test_images/
|-- internal_test_masks/
|-- external_test_images/
`-- external_test_masks/
```

Each image and mask pair must share the same file stem, for example:

```text
trainval_images/0001.nii.gz
trainval_masks/0001.nii.gz
```

Supported file suffixes by default:

- `.nii.gz`
- `.nii`
- `.nii(1).gz`

If your filenames differ, pass a custom list with `--nifti_exts`.

## Required Metadata Files

Place these CSV files under `./data/`.

### `internal_group.csv`

Required columns:

- `ID`
- `label`
- `group`

Example:

| ID   | label | group |
|------|------:|-------|
| 0001 | 0     | train |
| 0002 | 1     | train |
| 0101 | 0     | test  |

### `external_label.csv`

Required columns:

- `ID`
- `label`

Example:

| ID   | label |
|------|------:|
| E001 | 0     |
| E002 | 1     |

## Optional Grouping Column

If you want patient-level cross-validation, add a grouping column such as `patient_id` to `internal_group.csv` and run training with:

```bash
python train_fusion_mamba.py --group_col patient_id
```

When the column is present, the training pipeline will try to keep the same patient out of both fold-train and fold-val splits.

## Output Directories

The pipeline creates outputs under `./outputs/`:

```text
outputs/
|-- radiomics_features/
|-- selected_features/
|-- checkpoints/
`-- logs/
```

These directories are ignored by Git and should stay local.

## Pre-Run Checklist

Before training:

- every `ID` in the CSV matches an image filename stem
- every ROI image has a matching ROI mask
- `internal_group.csv` contains both `train` and `test`
- labels are binary integers
- train, internal-test, and external-test folders are separated correctly

## Common Problems

1. ID mismatch

CSV uses `0001`, but the file is named `001.nii.gz`. Rename files or fix the CSV.

2. Wrong ROI label in masks

PyRadiomics assumes the ROI label is `1` by default. Adjust extraction settings if your masks use another label.

3. Inconsistent split placement

Cases marked as `group=test` must be stored in `internal_test_images/` and `internal_test_masks/`, not in the train/val folders.
