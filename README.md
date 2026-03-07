# ESCC-LNM

Official code for a 3D CT plus radiomics fusion pipeline built around a Mamba-based image encoder and gated residual fusion.

This repository is prepared for public release:

- No patient data is included.
- Default paths are relative and Git-safe.
- Training uses a locked paper configuration by default.
- Radiomics preprocessing is isolated within each fold to reduce leakage.
- Training writes `run_metadata.json` for reproducibility.

## Repository Layout

```text
.
|-- run_standard_radiomics.py
|-- select_radiomics_features.py
|-- data_pipeline_fusion.py
|-- mamba_fusion_model.py
|-- train_fusion_mamba.py
|-- README.md
|-- README_DATA.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
```

Runtime outputs are written under `./outputs/` and are ignored by Git.

## Environment

Create a clean environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:

- Install the correct PyTorch build for your CUDA setup.
- `mamba-ssm` must be compatible with your local PyTorch and CUDA versions.
- For final paper results, archive the exact environment with `pip freeze > outputs/environment_freeze.txt`.

## Data Layout

Place your private dataset under `./data/`:

```text
data/
|-- trainval_images/
|-- trainval_masks/
|-- internal_test_images/
|-- internal_test_masks/
|-- external_test_images/
|-- external_test_masks/
|-- internal_group.csv
`-- external_label.csv
```

Metadata requirements:

- `internal_group.csv`: columns `ID`, `label`, `group`
- `external_label.csv`: columns `ID`, `label`
- `group` should be `train` or `test`
- `ID` must match the image and mask filename stem

See [README_DATA.md](README_DATA.md) for full details.

## Workflow

### 1. Extract radiomics

```bash
python run_standard_radiomics.py \
  --data_dir ./data \
  --output_dir ./outputs/radiomics_features \
  --skip_existing
```

This step only extracts features. It does not perform feature selection.

### 2. Select radiomics features

Recommended for cross-validation reporting:

```bash
python select_radiomics_features.py \
  --mode fold \
  --fold_idx 0 \
  --n_splits 5 \
  --internal_csv ./data/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features
```

Repeat `--fold_idx 0..4` to generate `fold01` to `fold05`.

Simpler first run:

```bash
python select_radiomics_features.py \
  --mode global \
  --internal_csv ./data/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features/global
```

### 3. Train and evaluate

Fold-specific selected radiomics:

```bash
python train_fusion_mamba.py \
  --output_dir ./outputs/checkpoints \
  --rad_root_dir ./outputs/selected_features \
  --internal_csv ./data/internal_group.csv \
  --external_csv ./data/external_label.csv \
  --group_col patient_id
```

Global selected radiomics:

```bash
python train_fusion_mamba.py \
  --output_dir ./outputs/checkpoints \
  --rad_trainval_csv ./outputs/selected_features/global/radiomics_internal_trainval_sel.csv \
  --rad_internal_test_csv ./outputs/selected_features/global/radiomics_internal_test_sel.csv \
  --rad_external_test_csv ./outputs/selected_features/global/radiomics_external_test_sel.csv \
  --internal_csv ./data/internal_group.csv \
  --external_csv ./data/external_label.csv
```

By default, training uses the locked paper configuration in `FINAL_CFG`. Use `--unlocked` only for ablation or exploratory runs.

## Leakage Control

The codebase includes these safeguards:

- Fold-isolated median imputation and standardization in `data_pipeline_fusion.py`
- Optional patient-level grouped cross-validation with `--group_col`
- Fold-specific radiomics selection with `select_radiomics_features.py --mode fold`
- Held-out internal and external test evaluation from the best validation checkpoint

## Reproducibility

Each training run records:

- training log at `outputs/logs/train_fusion_mamba.log`
- effective runtime arguments and environment in `outputs/checkpoints/run_metadata.json`
- fold metrics CSVs in `outputs/checkpoints/`

For paper submission, keep:

- exact commands used for each experiment
- exported environment package list
- selected feature lists per fold
- random seed and data split metadata

## Git Release Checklist

Before pushing:

- confirm `data/`, `data3D/`, and `outputs/` are not tracked
- do not upload raw patient identifiers
- do not upload NIfTI, masks, logs, checkpoints, or extracted feature tables
- keep the README consistent with the actual default CLI arguments
- rerun a syntax check such as `python -m py_compile *.py`

## Citation

If you use this repository in academic work, cite the corresponding paper and describe any deviations from the released locked configuration.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
