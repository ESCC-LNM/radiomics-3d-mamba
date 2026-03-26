# Data Requirements and Schema

The final multimodal fusion pipeline expects three aligned modalities per retained sample:

1. a cropped 3D CT volume
2. a corresponding 3D binary segmentation mask
3. a row in a selected radiomics feature table

## Cohort organisation

The final pipeline uses:

- a pooled development cohort for training
- an internal held-out test cohort
- an external held-out test cohort

If the optional preliminary cross-validation workflow is used, only the development cohort enters 5-fold cross-validation. Internal and external held-out cohorts must remain untouched during that stage.

## Metadata tables

The public scripts are configuration-driven through `configs/study_config.template.json`.

### Internal metadata CSV

The internal metadata CSV must provide at least these semantics:

- `sample_id`: unique imaging sample identifier
- `patient_id`: patient identifier used for patient-level splitting
- `label`: binary ground-truth label
- `group`: cohort role column that separates development rows from internal held-out rows

The default template maps these semantics to the following public column names:

- `sample_id -> ID`
- `patient_id -> patient_id`
- `label -> label`
- `group -> group`

### External metadata CSV

The external metadata CSV must provide at least:

- `sample_id`
- `label`
- `external_patient_id` if you want patient-level identifiers carried into exported predictions

The public example files are:

- `examples/metadata/internal_metadata.example.csv`
- `examples/metadata/external_metadata.example.csv`

## Imaging files

Provide separate local directories for:

- development CT crops
- internal held-out CT crops
- external held-out CT crops
- development masks
- internal held-out masks
- external held-out masks

Sample identifiers are resolved from file names after removing suffixes such as `.nii.gz`, `.nii`, or `.nii(1).gz`.

For the final fusion model, a retained sample must have all four of the following:

- a metadata row
- an image file
- a mask file
- a selected radiomics row

Samples missing any required element are skipped, and the final data-pipeline log reports how many rows were dropped per cohort.

## Radiomics tables

The manuscript-aligned final script `training/train_final_multimodal_mamba_fusion.py` expects three selected radiomics CSV tables:

- development selected table
- internal test selected table
- external test selected table

The default config keys are:

- `paths.final_selected_radiomics.development_csv`
- `paths.final_selected_radiomics.internal_test_csv`
- `paths.final_selected_radiomics.external_test_csv`

Each selected table must contain:

- an identifier column named `ID`, or an equivalent first column that can be renamed to `ID`
- the same selected radiomics feature columns in the same semantic feature space

The optional preprocessing workflow writes these files by default under:

- `<selected_features_root>/final/development_selected.csv`
- `<selected_features_root>/final/internal_test_selected.csv`
- `<selected_features_root>/final/external_test_selected.csv`


## Public release recommendation

For publication-facing release, include:

- this schema document
- a non-sensitive example configuration file
- example metadata illustrating the required column structure
- a clear statement about which raw data cannot be redistributed
