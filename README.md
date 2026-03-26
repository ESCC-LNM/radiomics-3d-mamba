# ESCC LNM Multimodal Prediction Repository

This directory contains the manuscript-aligned public code release for esophageal squamous cell carcinoma (ESCC) lymph node metastasis (LNM) prediction using radiomics and 3D deep-learning models.

## Public release scope

The public package currently includes two kinds of material:

1. End-to-end runnable scripts for the manuscript-aligned multimodal fusion pipeline and the optional preliminary 5-fold development workflow.
2. Architecture modules for comparator models that are released as code definitions only.

### Runnable command-line entry points

- `training/train_final_multimodal_mamba_fusion.py`: final manuscript-aligned multimodal training on the pooled development cohort with held-out internal and external evaluation
- `preprocessing/generate_patient_split_manifest.py`: patient-level cohort and outer-fold manifest generation for optional preliminary tuning
- `preprocessing/extract_radiomics_features.py`: radiomics extraction only
- `preprocessing/select_supervised_radiomics_features.py`: supervised radiomics feature selection
- `training/train_preliminary_multimodal_mamba_fusion.py`: optional preliminary outer cross-validation for model-selection purposes only
- `release/validate_release_integrity.py`: lightweight release sanity check for required files and configuration structure

### Architecture modules only

The following files define model components or comparator architectures, but they are not standalone manuscript-locked training CLI scripts in this public release:

- `models/model_radiomics_mlp.py`
- `models/model_image_resnet34.py`
- `models/model_image_swin.py`
- `models/model_image_mamba.py`
- `models/model_multimodal_resnet34_fusion.py`
- `models/model_multimodal_swin_fusion.py`
- `models/model_multimodal_mamba_fusion.py`

### Package layout

- `training/`: manuscript-aligned and preliminary training entry points
- `preprocessing/`: cohort manifest generation and radiomics preprocessing scripts
- `data/`: dataloader construction modules
- `models/`: model-definition modules
- `utils/`: shared helper utilities
- `release/`: repository validation utilities

## Final manuscript-aligned model

The final multimodal model is a dual-branch radiomics-3D Mamba fusion model with learnable gated logit fusion:

```text
fusion = (1 - alpha) * logits_rad + alpha * logits_img
```

Here `alpha` is a sample-wise scalar gate estimated from the concatenated radiomics and image features.

## Methodological scope

The public code is aligned with the manuscript wording used for submission:

- image input for the final fusion model is dual-channel: `[CT, binary mask]`
- radiomics input is a pre-extracted, preselected feature vector loaded from selected CSV tables
- radiomics preprocessing uses training-set-only median imputation and `StandardScaler` fitting, then applies the fitted parameters unidirectionally to held-out cohorts
- final fusion training uses a fixed epoch count, no validation split, no early stopping, and a fixed decision threshold of `0.5`
- optional 5-fold cross-validation is for preliminary hyperparameter or epoch selection only, not for final held-out performance reporting

See `MANUSCRIPT_ALIGNMENT.md` for a direct mapping between paper claims and implementation files.

## Configuration

Use `configs/study_config.template.json` as the starting point for all public commands. The template is manuscript-aligned in terms of hyperparameters and training policy, but it intentionally contains placeholder local paths that must be filled on your machine.

Important path groups to edit before running the final pipeline:

- `paths.internal_metadata_csv`
- `paths.external_metadata_csv`
- `paths.images.*`
- `paths.masks.*`
- `paths.selected_features_root`
- `paths.final_selected_radiomics.*`
- `paths.final_output_dir`

## Recommended execution order

### Path A: final manuscript-aligned training and held-out evaluation

1. Prepare metadata tables and local data directories as described in `README_DATA.md`.
2. Provide the final selected radiomics CSV tables listed under `paths.final_selected_radiomics` in the config.
3. Fill `configs/study_config.template.json` with your local paths.
4. Run:

```bash
python -m training.train_final_multimodal_mamba_fusion --config configs/study_config.template.json
```

### Path B: optional preliminary 5-fold tuning before final retraining

1. Fill `configs/study_config.template.json` with your local paths.
2. Run:

```bash
python -m preprocessing.generate_patient_split_manifest --config configs/study_config.template.json
python -m preprocessing.extract_radiomics_features --config configs/study_config.template.json
python -m preprocessing.select_supervised_radiomics_features --config configs/study_config.template.json
python -m training.train_preliminary_multimodal_mamba_fusion --config configs/study_config.template.json
```

3. Lock the chosen hyperparameters or epoch count externally, then execute the final manuscript-aligned training script shown in Path A.

## What is not included

- raw CT images, masks, or clinical metadata
- a filled manuscript-private configuration file with site-specific local paths
- standalone final CLI trainers for the radiomics-only and image-only comparator models

## Data access and privacy

Clinical images, masks, and metadata are not included in this repository. Public users should prepare local data according to `README_DATA.md` and the example metadata files in `examples/metadata/`.

## Environment

Install the required Python packages listed in `requirements.txt`. The `mamba-ssm` backend requires a CUDA-capable environment for actual training and inference with the final fusion model.

## License

This repository is distributed under the MIT License. See the repository-root `LICENSE` file.
