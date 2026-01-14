# Data Preparation Guide (README_DATA.md)

This repository **does not include any data**.  
You must prepare your own dataset and place it under `./data/` following the conventions below.

> ⚠️ Privacy reminder:
> - Do **NOT** upload patient identifiers to GitHub.
> - Do **NOT** push `data/`, `outputs/`, logs, checkpoints, or extracted radiomics tables.

---

## 1) Required folder structure

Create the following directories under `./data/`:

```
data/
├── trainval_images/           # internal train/val ROI images (NIfTI)
├── trainval_masks/            # internal train/val ROI masks  (NIfTI)
├── internal_test_images/      # internal test ROI images      (NIfTI)
├── internal_test_masks/       # internal test ROI masks       (NIfTI)
├── external_test_images/      # external test ROI images      (NIfTI)
└── external_test_masks/       # external test ROI masks       (NIfTI)
```

### File naming rule

Each ROI image must match its ROI mask by the **same ID stem**.

Example:

```
trainval_images/0001.nii.gz
trainval_masks/0001.nii.gz
```

Supported extensions (default):
- `.nii.gz`
- `.nii`
- `.nii(1).gz`

You can change the extension list via `--nifti_exts` in `data_pipeline_fusion.py` / `train_fusion_mamba.py`.

---

## 2) Required metadata CSVs

Place these two CSV files under `./data/`:

### A) `internal_group.csv`

Must contain columns:

- `ID`   : case identifier (must match NIfTI filename stem)
- `label`: class label (0/1)
- `group`: `"train"` or `"test"`

Minimal example:

| ID   | label | group |
|------|------:|-------|
| 0001 | 0     | train |
| 0002 | 1     | train |
| 0101 | 0     | test  |

### B) `external_label.csv`

Must contain columns:

- `ID`   : case identifier (must match NIfTI filename stem)
- `label`: class label (0/1)

Minimal example:

| ID   | label |
|------|------:|
| E001 | 0     |
| E002 | 1     |

---

## 3) Output folders (auto-generated)

The pipeline will generate outputs under `./outputs/` (recommended):

```
outputs/
├── radiomics_features/    # Step 1 output (raw radiomics tables)
├── selected_features/     # Step 2 output (selected features)
├── checkpoints/           # Step 5 output (model weights)
└── logs/                  # logs (if configured)
```

These folders are **ignored by git** by default. Do not upload them.

---

## 4) Quick sanity checklist

Before running training, confirm:

- [ ] Every case in CSV has a matching image file in the corresponding folder.
- [ ] Every case has a matching mask file with the same ID.
- [ ] `internal_group.csv` contains both `train` and `test` groups.
- [ ] `label` values are integers (0/1).
- [ ] External data uses **external** folders and `external_label.csv`.

---

## 5) Common pitfalls

1) **ID mismatch**  
   Example: CSV uses `0001` but file is named `001.nii.gz`.  
   Fix: rename files or update CSV IDs.

2) **Mask label not equal to 1**  
   PyRadiomics expects the target ROI label is `1` by default.  
   If your masks use another label value, adjust radiomics extraction settings.

3) **Non-ROI images**  
   This code expects ROI-cropped 3D volumes (e.g., 32×64×64).  
   If your ROI size differs, update `--roi_size` consistently.

---

## Contact / Issues

If you encounter issues, please open a GitHub issue and include:
- your command
- environment info (Python version, OS, GPU/CUDA)
- the full stack trace

⚠️ Do **NOT** upload any data, patient IDs, or private paths in issues.
