# Data Requirements and Schema

This repository expects three data modalities for the multimodal fusion model:

1. a cropped 3D CT volume
2. a corresponding 3D binary segmentation mask
3. a row in a pre-extracted radiomics feature table

## Cohort organisation

The manuscript-aligned final pipeline uses:

- a pooled global training set
- an internal test set
- an external test set

If the optional preliminary cross-validation utilities are used, only the development cohort enters 5-fold cross-validation. Internal and external held-out cohorts must remain untouched during that stage.

## Required metadata columns

Final scripts (`train_final_mamba.py`, `train_final_radiomics.py`) expect CSV tables equivalent to the following semantics:

- `ID`: unique imaging sample identifier
- `label`: binary ground-truth label
- `group` (internal metadata only): values that separate `train` and `test`

The optional configuration-driven preprocessing utilities use a more explicit schema described by the keys in `configs/study_config.template.json`, including:

- `sample_id`
- `patient_id`
- `label`
- `group`
- optional external patient identifier

## Imaging files

For the final manuscript-aligned fusion pipeline, provide separate directories for:

- development / train-and-validation CT crops
- internal held-out CT crops
- external held-out CT crops
- development / train-and-validation masks
- internal held-out masks
- external held-out masks

Sample identifiers are resolved from file names after removing suffixes such as `.nii.gz`, `.nii`, or `.nii(1).gz`.

The fusion model assumes that a mask can be found for each retained imaging sample. Samples missing any of the three required elements (image, mask, radiomics row) are excluded from fusion-model training and evaluation.

## Radiomics tables

The manuscript-aligned code assumes that radiomics extraction and feature selection have already been completed before final model training.

The final fusion and radiomics-only scripts expect three selected radiomics CSV tables:

- global train/development table
- internal test table
- external test table

These tables must share the same selected feature columns, with an identifier column named `ID` or an equivalent first column that can be renamed to `ID`.

## Normalisation policy

The final manuscript-aligned pipeline fits median imputation and `StandardScaler` **only on the pooled global training set**, then applies the fitted preprocessing parameters to the internal and external test sets without refitting.

## Public release recommendation

For publication-facing release, include:

- this schema document
- a non-sensitive example configuration file
- example or simulated metadata illustrating the required column structure
- a clear statement about which raw data cannot be redistributed
