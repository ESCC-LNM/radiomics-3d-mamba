# ESCC-LNM

This repository provides the official code for our paper on **3D CT + Radiomics fusion** using a **Mamba-based sequence encoder** and a **gated residual fusion** strategy.

> **Data is NOT included.** You must prepare your own dataset following the folder/CSV conventions below.

---

## Highlights

- **Multi-modal fusion**: 3D ROI (NIfTI) + radiomics feature vectors.
- **Mamba backbone**: CNN → token sequence → Mamba blocks → image embedding.
- **Gated fusion**: radiomics baseline logits + gated residual delta logits (uncertainty-aware).
- **Leakage-aware pipeline**:
  - **Patient-level grouping (optional but recommended):** when `--group_col` (default `patient_id`) exists in `internal_group.csv`, CV is group-aware so the same patient never appears in both fold-train and fold-val.
  - **Fold-isolated radiomics preprocessing:** median imputation + standardization are **fit on fold-train only** in `data_pipeline_fusion.py`, then applied to val/internal/external.
  - **Leakage-safe feature selection:** `select_radiomics_features.py --mode fold` performs selection **within each fold-train** (recommended when reporting CV).
  - **Strict test-set isolation (recommended default):** internal/external test sets are evaluated **once after training** using the best validation checkpoint; enable `--eval_tests_during_train` only for debugging.

---

## Repository Structure

```text
My-Mamba-Radiomics-Paper/
│
├── run_standard_radiomics.py        # Step 1: radiomics extraction (PyRadiomics)
├── select_radiomics_features.py     # Step 2: feature selection (supports fold mode)
├── data_pipeline_fusion.py          # Step 3: dataloaders + fold-isolated preprocessing
├── mamba_fusion_model.py            # Step 4: model definition (Mamba + fusion)
├── train_fusion_mamba.py            # Step 5: training (5-fold CV)
│
├── data/                            # EMPTY in this repo (user provides data)
├── outputs/                         # training outputs (ignored by git)
│   ├── radiomics_features/
│   ├── selected_features/
│   ├── checkpoints/
│   └── logs/
│
├── requirements.txt
└── .gitignore
```

---

## Environment Setup

### 1) Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> If you use GPU, please install the correct **PyTorch + CUDA** build for your system.

**Dependencies (from `requirements.txt`):**
- numpy, pandas, scikit-learn, tqdm
- torch (>=2.0)
- monai, SimpleITK
- pyradiomics
- mamba-ssm

---

## Data Preparation

**This repo does not include any data.** Put your dataset under `./data/` and follow the structure below.

### 1) Required folder structure

```text
data/
├── trainval_images/           # internal train/val ROI images (NIfTI)
├── trainval_masks/            # internal train/val ROI masks  (NIfTI)
├── internal_test_images/      # internal test ROI images      (NIfTI)
├── internal_test_masks/       # internal test ROI masks       (NIfTI)
├── external_test_images/      # external test ROI images      (NIfTI)
└── external_test_masks/       # external test ROI masks       (NIfTI)
```

- Each ROI image must match its ROI mask by the **same ID stem** (e.g., `0001.nii.gz`).
- Supported extensions (default): `.nii.gz`, `.nii`, `.nii(1).gz` (configurable via `--nifti_exts`).

### 2) Required metadata CSVs

Place these under `./data/`:

- `internal_group.csv` must contain columns: `ID`, `label`, `group`
  - `group` must be `"train"` (train/val pool) or `"test"` (internal test set)
- `external_label.csv` must contain columns: `ID`, `label`

> Privacy reminder: **do NOT upload patient identifiers** or any data/outputs/logs/checkpoints to GitHub.

---

## Pipeline Usage (5 Steps)

### Step 1 — Extract radiomics features (PyRadiomics)

This script **only extracts features** (no feature selection). By default it **does NOT write labels** into output CSVs (safer).

```bash
python run_standard_radiomics.py \
  --data_dir ./data \
  --output_dir ./outputs/radiomics_features \
  --skip_existing
```

Outputs (default):
- `outputs/radiomics_features/radiomics_internal_trainval.csv`
- `outputs/radiomics_features/radiomics_internal_test.csv`
- `outputs/radiomics_features/radiomics_external_test.csv`

**Optional privacy controls:**
- `--id_mode hash --id_salt YOUR_SALT` (adds `anon_id`, hashes IDs in logs)
- `--hash_id_in_csv` (**not recommended**) replaces `ID` in CSV and can break downstream ID matching

---

### Step 2 — Select radiomics features (important for leakage control)

This step generates the **selected** radiomics CSVs used by training.

#### Option A (Recommended for reporting K-Fold CV): **fold mode**

Fold mode selects features using **only the fold-train subset**, reducing feature-selection leakage.

Example (fold 1, i.e., `fold_idx=0`):

```bash
python select_radiomics_features.py \
  --mode fold --n_splits 5 --fold_idx 0 \
  --internal_csv ./data/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features
```

This produces:
- `outputs/selected_features/fold01/radiomics_internal_trainval_sel.csv`
- `outputs/selected_features/fold01/radiomics_internal_test_sel.csv`
- `outputs/selected_features/fold01/radiomics_external_test_sel.csv`
- `outputs/selected_features/fold01/selected_features.txt`

Repeat for `fold_idx=1..4` (fold02..fold05).

#### Option B (Easiest first run): **global mode**

Global mode selects one fixed feature set using the **entire internal train pool**.  
**Warning:** if you later report K-Fold validation metrics from the same pool, global selection can inflate CV performance.

```bash
python select_radiomics_features.py \
  --mode global \
  --internal_csv ./data/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features/global
```

---

### Step 3 — Data loading (fold-isolated preprocessing)

`data_pipeline_fusion.py` is called by the trainer. It performs:
- **Group-aware** cross-validation on the internal training pool:
  - uses `StratifiedGroupKFold` (if available) or `GroupKFold` when `--group_col` (default: `patient_id`) exists in `internal_group.csv`
  - falls back to sample-level `StratifiedKFold` if the grouping column is missing (a warning will be printed)
- radiomics preprocessing (median imputation + standardization) **fit ONLY on fold-train**, then applied to val/internal/external
- MONAI transforms for 3D images (windowing + resize to fixed `roi_size`)
- privacy-safe defaults (IDs not returned unless explicitly enabled)

No separate command is required here.


---

### Step 4 — Model

`mamba_fusion_model.py` defines:
- Mamba-based image encoder
- radiomics MLP branch
- gated residual fusion

No separate command is required here.

---

### Step 5 — Train & evaluate (5-Fold CV)

The trainer runs 5-fold CV and reports metrics. It uses a **final locked config** for reproducibility (paper release).

#### A) Quick start (recommended, leakage-safe)

This quick start assumes you will **report 5-fold cross-validation** results from the internal training pool.
To prevent feature-selection leakage, generate **fold-specific** selected radiomics for each fold, then train with
`--rad_root_dir` so the trainer auto-loads the matching `foldXX/` CSVs.

1) **Fold-specific radiomics selection (fold01..fold05)**

```bash
for i in 0 1 2 3 4; do
  python select_radiomics_features.py \
    --mode fold \
    --fold_idx $i \
    --internal_csv ./data/internal_group.csv \
    --radiomics_dir ./outputs/radiomics_features \
    --out_dir ./outputs/selected_features
done
```

2) **Train (5-fold CV) with strict test-set isolation (recommended default)**

```bash
python train_fusion_mamba.py \
  --output_dir ./outputs/checkpoints \
  --rad_root_dir ./outputs/selected_features \
  --internal_csv ./data/internal_group.csv \
  --external_csv ./data/external_label.csv \
  --group_col patient_id
```

> Notes:
> - `--group_col patient_id` enforces **patient-level grouping** during CV *if* the `patient_id` column exists in `internal_group.csv`.
>   If the column is missing, the code falls back to sample-level CV and prints a warning.
> - By default, **internal/external test sets are evaluated only once after training** (best checkpoint chosen by validation AUC).
>   If you explicitly want the legacy behavior, set `--eval_tests_during_train` (not recommended for strict “sealed” external testing).

#### B) Correct pairing for fold-mode features (important!)


If you used Step 2 **fold mode**, training must read the matching `foldXX/` selected CSVs **for the same fold**.

This repo supports two ways:

**Way 1 (recommended): use `--rad_root_dir` and let the trainer auto-pick per fold**

```bash
python train_fusion_mamba.py \
  --rad_root_dir ./outputs/selected_features \
  --output_dir ./outputs/checkpoints
```

- It will look for `./outputs/selected_features/fold01/` ... `fold05/` and automatically use the correct CSVs inside each fold directory.
- If you want to run just one fold:

```bash
python train_fusion_mamba.py \
  --run_fold 1 \
  --rad_root_dir ./outputs/selected_features \
  --output_dir ./outputs/checkpoints
```

**Way 2: manual per-fold CSV paths**

```bash
for k in 1 2 3 4 5; do
  python train_fusion_mamba.py \
    --run_fold ${k} \
    --output_dir ./outputs/checkpoints \
    --rad_trainval_csv ./outputs/selected_features/fold$(printf "%02d" ${k})/radiomics_internal_trainval_sel.csv \
    --rad_internal_test_csv ./outputs/selected_features/fold$(printf "%02d" ${k})/radiomics_internal_test_sel.csv \
    --rad_external_test_csv ./outputs/selected_features/fold$(printf "%02d" ${k})/radiomics_external_test_sel.csv
done
```
---

## Leakage & Privacy Notes (Please Read)

### 1) Cross-validation leakage

To keep CV fair:
- radiomics preprocessing (imputer/scaler) must be fit on **fold-train only** (done in `data_pipeline_fusion.py`)
- radiomics **feature selection** should be fold-specific if you report CV validation metrics (use `select_radiomics_features.py --mode fold`)

### 2) Sensitive identifiers

This repo is designed to avoid accidental leakage by default:
- pipeline does **not return IDs** unless explicitly enabled
- logging avoids printing raw IDs by default
- never push `data/`, `outputs/`, logs, checkpoints, or radiomics tables to GitHub

---

## Outputs

By default:
- Step 1: `outputs/radiomics_features/`
- Step 2: `outputs/selected_features/`
- Step 5: `outputs/checkpoints/` and `outputs/logs/`

These should be ignored by git via `.gitignore`.

---

## Common Pitfalls

- **ID mismatch**: CSV has `0001` but file is `001.nii.gz` → rename files or update CSV IDs.
- **Mask label not equal to 1**: PyRadiomics default ROI label is `1`; if your mask uses another label value, adjust extraction settings.
- **ROI size mismatch**: this pipeline expects ROI-cropped 3D volumes (e.g., 32×64×64). If different, update `--roi_size` consistently.

---


