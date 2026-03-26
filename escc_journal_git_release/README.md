# ESCC LNM Multimodal Prediction Repository

This repository contains the manuscript-aligned codebase for esophageal squamous cell carcinoma (ESCC) lymph node metastasis (LNM) prediction using radiomics and 3D deep-learning models.

The public release is organised around two layers of analysis:

1. **Final manuscript analysis**: fixed-protocol training and held-out evaluation scripts that correspond to the paper-facing methods text.
2. **Optional preliminary development utilities**: patient-level cross-validation tools that can be used for hyperparameter selection before the final fixed-protocol retraining stage.

The final manuscript-aligned fusion model is a **dual-branch radiomics–3D Mamba model with learnable gated logit fusion**. Its image branch uses a 3D convolutional feature extractor followed by stacked Mamba blocks, and its radiomics branch uses a single-hidden-layer MLP. The final fusion logits are computed as:

```text
fusion = (1 - α) * logits_rad + α * logits_img
```

where `α` is a sample-wise scalar gate estimated from the concatenated radiomics and image features.

## What this repository is meant to support

### Final manuscript-aligned scripts

- `data_pipeline_final.py` — final dataloaders for the pooled global training set, internal test set, and external test set
- `mamba_fusion_model.py` — radiomics–3D Mamba fusion model used in the final manuscript-aligned implementation
- `train_final_mamba.py` — fixed-protocol final training and held-out evaluation for the fusion model
- `train_final_radiomics.py` — fixed-protocol radiomics-only baseline
- `model_resnet34.py`, `model_swin.py`, `model_mamba.py` — image-only baseline encoders
- `resnet_fusion_model.py`, `st_fusion_model.py` — fusion-model baseline variants with alternative image encoders

### Optional preliminary model-selection utilities

- `make_split_manifest.py` — freezes patient-level cohort roles and outer-fold assignments
- `run_standard_radiomics.py` — unsupervised radiomics extraction
- `select_radiomics_features.py` — supervised radiomics feature selection
- `data_pipeline_fusion.py` — development-stage dataloaders for optional 5-fold tuning
- `train_fusion_mamba.py` — development-stage outer cross-validation for preliminary hyperparameter/epoch selection
- `experiment_utils.py` — shared helpers

## Manuscript-aligned methodological scope

The repository is aligned to the manuscript wording adopted for submission:

- image input for the fusion model is **dual-channel**: `[CT, binary mask]`
- radiomics input is a **pre-extracted, preselected 112-dimensional feature vector**
- radiomics preprocessing uses **training-set-only median imputation and StandardScaler fitting**, then applies the fitted parameters unidirectionally to held-out cohorts
- final fusion training uses **fixed 220 epochs**, **no validation split**, **no early stopping**, and **a fixed decision threshold of 0.5**
- preliminary 5-fold cross-validation, when used, is for **hyperparameter and epoch selection only**, not for final test-set performance estimation

See `MANUSCRIPT_ALIGNMENT.md` for a direct mapping between paper statements and implementation files.

## Recommended execution order

The practical order depends on whether you want only the final manuscript-aligned pipeline or also the optional preliminary tuning stage.

### Path A — final manuscript-aligned training/evaluation

1. Prepare the metadata tables and directories described in `README_DATA.md`.
2. Generate or provide the selected radiomics CSV tables.
3. Run `train_final_mamba.py` for the multimodal model.
4. Run `train_final_radiomics.py` and any desired baseline models for comparative analysis.

### Path B — optional preliminary 5-fold tuning before final retraining

1. `python make_split_manifest.py --config configs/manuscript_locked_template.json`
2. `python run_standard_radiomics.py --config configs/manuscript_locked_template.json`
3. `python select_radiomics_features.py --config configs/manuscript_locked_template.json`
4. `python train_fusion_mamba.py --config configs/manuscript_locked_template.json`
5. Lock the chosen hyperparameters/epoch count, then execute the final manuscript-aligned training scripts.

## Journal-facing release checklist

Before public upload, confirm the following:

- remove or replace all site-specific file-system paths
- provide a manuscript-locked configuration file or complete command log
- archive a repository snapshot associated with the submitted manuscript version
- document data-access restrictions and any non-public data dependencies
- confirm that the final reported numbers come from the fixed final scripts rather than from cross-validation folds

A detailed release checklist is provided in `REPRODUCIBILITY_CHECKLIST.md`.

## Data access and privacy

Clinical images, masks, and metadata are not included in this repository. Public users should prepare local data according to `README_DATA.md` and the example configuration templates in `configs/`.

## Environment

Install the required Python packages listed in `requirements.txt`. Because the Mamba backend depends on CUDA-enabled kernels, the fusion model requires a compatible CUDA environment for actual training and inference.

## Important note on licensing

A project should not be published to a public Git host without an explicit approved software license. This package includes `LICENSE_PENDING.md` as a placeholder so the release process does not overlook that step. Replace it with an institution-approved license before public publication.
