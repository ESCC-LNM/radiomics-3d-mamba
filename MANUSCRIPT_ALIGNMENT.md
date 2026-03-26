# Manuscript-to-Code Alignment Notes

This file documents how the manuscript wording maps onto the public repository contents.

## Final multimodal model

**Paper wording**

- dual-branch radiomics-3D Mamba model with learnable gated fusion
- input consists of pre-extracted radiomics features and 3D image features
- CT and binary mask are used together for the image branch
- fusion is sample-wise and applied at the logit level

**Implementation anchors**

- `models/model_multimodal_mamba_fusion.py`
- `data/data_loading_final_multimodal.py`
- `training/train_final_multimodal_mamba_fusion.py`
- `configs/study_config.template.json`

## Radiomics branch

**Paper wording**

- pre-extracted, preselected radiomics feature vector
- single-hidden-layer MLP with ReLU and Dropout
- branch-specific radiomics classification head

**Implementation anchors**

- `models/model_multimodal_mamba_fusion.py`
- `models/model_radiomics_mlp.py` for the standalone radiomics MLP architecture definition

## Image branch

**Paper wording**

- custom 3D convolutional feature extractor followed by stacked Mamba blocks
- global pooling to obtain image-level features
- branch-specific image classification head

**Implementation anchors**

- `models/model_multimodal_mamba_fusion.py`
- `models/model_image_mamba.py` for the image-only comparator backbone definition

## Final training protocol

**Paper wording**

- fixed epoch count
- no validation split
- no early stopping
- fixed threshold of `0.5`
- training-set-only radiomics preprocessing applied unidirectionally to held-out cohorts

**Implementation anchors**

- `training/train_final_multimodal_mamba_fusion.py`
- `data/data_loading_final_multimodal.py`
- `configs/study_config.template.json`

## Optional preliminary 5-fold tuning

**Paper wording**

- optional preliminary 5-fold stratified cross-validation used only for parameter or epoch selection before final retraining

**Implementation anchors**

- `preprocessing/generate_patient_split_manifest.py`
- `preprocessing/extract_radiomics_features.py`
- `preprocessing/select_supervised_radiomics_features.py`
- `data/data_loading_cv_multimodal.py`
- `training/train_preliminary_multimodal_mamba_fusion.py`

## Comparator-model release status

The public repository includes comparator model definitions, but it does not currently include manuscript-locked final training CLI drivers for each comparator architecture. The released comparator files should therefore be read as architecture references rather than fully packaged reproduction commands.

Files in this category include:

- `models/model_radiomics_mlp.py`
- `models/model_image_resnet34.py`
- `models/model_image_swin.py`
- `models/model_image_mamba.py`
- `models/model_multimodal_resnet34_fusion.py`
- `models/model_multimodal_swin_fusion.py`

## Deliberate exclusions from the public code text

The manuscript should not over-claim any of the following unless separate code or analysis artefacts are released alongside this repository:

- calibration analysis
- decision-curve analysis
- DeLong testing
- subgroup analyses
- manuscript-locked comparator training scripts that are not present in this package
