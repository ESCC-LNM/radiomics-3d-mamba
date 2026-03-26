# Manuscript-to-Code Alignment Notes

This file documents the intended alignment between the manuscript wording and the repository implementation.

## Final multimodal model

**Paper wording**
- dual-branch radiomics–3D Mamba model with learnable gated fusion
- input consists of pre-extracted radiomics features and 3D image features
- CT and binary mask are used together for the image branch
- fusion is sample-wise gated at the logit level

**Implementation anchors**
- `mamba_fusion_model.py`
- `data_pipeline_final.py`
- `train_final_mamba.py`

## Radiomics branch

**Paper wording**
- 112-dimensional pre-extracted, preselected radiomics feature vector
- single-hidden-layer MLP with ReLU and Dropout
- branch-specific radiomics classification head

**Implementation anchors**
- `mamba_fusion_model.py`
- `train_final_radiomics.py`

## Image branch

**Paper wording**
- custom 3D convolutional feature extractor followed by two stacked Mamba blocks
- adaptive/global average pooling to obtain image-level features
- branch-specific image classification head

**Implementation anchors**
- `mamba_fusion_model.py`

## Final training protocol

**Paper wording**
- fixed 220 epochs
- no validation split
- no early stopping
- fixed threshold of 0.5
- training-set-only radiomics preprocessing applied unidirectionally to held-out cohorts

**Implementation anchors**
- `train_final_mamba.py`
- `train_final_radiomics.py`
- `data_pipeline_final.py`

## Preliminary 5-fold tuning

**Paper wording**
- optional preliminary 5-fold stratified cross-validation used only for parameter/epoch selection before final retraining

**Implementation anchors**
- `make_split_manifest.py`
- `data_pipeline_fusion.py`
- `train_fusion_mamba.py`

## Deliberate exclusions from the public code text

The manuscript should not over-claim any of the following unless there is separate code or analysis material included in the release:

- calibration analysis
- decision-curve analysis
- DeLong testing
- subgroup analyses
- any feature-extraction details not represented by the public radiomics scripts/configuration
