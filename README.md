# ESCC LNM Multimodal Prediction Repository

This repository contains the manuscript-aligned public code release for esophageal squamous cell carcinoma (ESCC) lymph node metastasis (LNM) prediction using radiomics and 3D deep-learning models. The project was developed to support transparent, reproducible, and publication-ready research on multimodal imaging analysis, with the final manuscript model centered on the integration of handcrafted radiomics descriptors and volumetric deep representations derived from contrast-enhanced CT.

The final manuscript-aligned framework is a **dual-branch radiomics-3D Mamba fusion model with learnable gated logit fusion**. The radiomics branch operates on pre-extracted and preselected radiomics feature vectors, whereas the image branch learns volumetric representations from **dual-channel inputs** composed of the CT volume and the corresponding binary mask. The two prediction streams are integrated through a sample-wise scalar gate:

```text
fusion = (1 - alpha) * logits_rad + alpha * logits_img
```

where `alpha` is estimated from the concatenated radiomics and image representations and is used to adaptively rebalance branch contributions on a per-case basis.

This public release is organized around **journal-level methodological transparency**. In particular, it distinguishes preliminary model-selection workflows from final held-out evaluation, documents the preprocessing and cohort-handling assumptions used in the study, and provides repository-level materials for manuscript-code alignment and reproducibility inspection.

---

## Clinical and technical motivation

Accurate preoperative prediction of LNM in ESCC is clinically important because nodal status directly affects staging, treatment planning, and prognosis assessment. Conventional image interpretation can be limited by inter-reader variability and by the difficulty of integrating subtle morphological, textural, and volumetric cues across heterogeneous lesions. This project addresses that problem by combining two complementary information streams:

- **Radiomics features**, which provide structured quantitative descriptors derived from lesion-centered regions of interest.
- **3D deep image features**, which enable end-to-end representation learning from volumetric CT data.

The repository is therefore designed not only to provide the final multimodal model, but also to make the modeling workflow sufficiently explicit for manuscript review, method inspection, and controlled reproduction.

---

## Project highlights

- Manuscript-aligned implementation of a **dual-branch radiomics-3D Mamba fusion model**.
- **Learnable gated logit fusion** between radiomics and image branches.
- Support for **dual-channel image input**: `[CT, binary mask]`.
- Training-set-only radiomics preprocessing, with held-out internal and external cohort isolation.
- Explicit separation between:
  - **optional preliminary 5-fold development workflow** for model-selection purposes only;
  - **final manuscript-aligned training and evaluation** on the pooled development cohort with internal and external held-out testing.
- Public-release materials for manuscript-code alignment, repository validation, and reproducibility review.

---

## Public release scope

The public package currently includes two kinds of material:

1. **End-to-end runnable scripts** for the manuscript-aligned multimodal fusion pipeline and the optional preliminary 5-fold development workflow.
2. **Architecture modules for comparator models** that are released as code definitions only.

### Runnable command-line entry points

- `training/train_final_multimodal_mamba_fusion.py`  
  Final manuscript-aligned multimodal training on the pooled development cohort with held-out internal and external evaluation.
- `preprocessing/generate_patient_split_manifest.py`  
  Patient-level cohort and outer-fold manifest generation for optional preliminary tuning.
- `preprocessing/extract_radiomics_features.py`  
  Radiomics extraction only.
- `preprocessing/select_supervised_radiomics_features.py`  
  Supervised radiomics feature selection.
- `training/train_preliminary_multimodal_mamba_fusion.py`  
  Optional preliminary outer cross-validation for model-selection purposes only.
- `release/validate_release_integrity.py`  
  Lightweight release sanity check for required files and configuration structure.

### Architecture modules only

The following files define model components or comparator architectures, but they are **not** standalone manuscript-locked training CLI scripts in this public release:

- `models/model_radiomics_mlp.py`
- `models/model_image_resnet34.py`
- `models/model_image_swin.py`
- `models/model_image_mamba.py`
- `models/model_multimodal_resnet34_fusion.py`
- `models/model_multimodal_swin_fusion.py`
- `models/model_multimodal_mamba_fusion.py`

---

## Final manuscript-aligned model

The final multimodal model is a **dual-branch radiomics-3D Mamba fusion model with learnable gated logit fusion**.

### Input definition

- **Radiomics branch input**: pre-extracted, preselected radiomics feature vectors loaded from selected CSV tables.
- **Image branch input**: dual-channel volumetric input composed of:
  - CT volume
  - corresponding binary mask

### Fusion rule

```text
fusion = (1 - alpha) * logits_rad + alpha * logits_img
```

Here, `alpha` is a sample-wise scalar gate estimated from the concatenated radiomics and image features.

### Modeling rationale

The radiomics branch provides structured handcrafted quantitative descriptors, whereas the 3D Mamba image branch learns volumetric contextual representations directly from image input. The gated fusion strategy allows the model to dynamically redistribute predictive emphasis between the two branches for each case rather than relying on a fixed global weighting scheme.

---

## Methodological scope

The public code is aligned with the manuscript wording used for submission. The following methodological constraints define the intended interpretation of this repository:

- image input for the final fusion model is **dual-channel**: `[CT, binary mask]`
- radiomics input is a **pre-extracted, preselected feature vector** loaded from selected CSV tables
- radiomics preprocessing uses **training-set-only median imputation** and **`StandardScaler` fitting**, then applies the fitted parameters unidirectionally to held-out cohorts
- final fusion training uses a **fixed epoch count**, **no validation split**, **no early stopping**, and a **fixed decision threshold of `0.5`**
- optional **5-fold cross-validation** is reserved for **preliminary hyperparameter or epoch selection only**, and is **not** used for final held-out performance reporting

See `MANUSCRIPT_ALIGNMENT.md` for a direct mapping between manuscript claims and implementation files.

---

## Repository layout

### Package layout

- `training/` — manuscript-aligned and preliminary training entry points
- `preprocessing/` — cohort manifest generation and radiomics preprocessing scripts
- `data/` — dataloader construction modules
- `models/` — model-definition modules
- `utils/` — shared helper utilities
- `release/` — repository validation utilities
- `configs/` — public configuration templates
- `examples/` — example metadata structures and release examples

### Release-oriented documentation

- `MANUSCRIPT_ALIGNMENT.md` — manuscript-to-code alignment notes
- `REPRODUCIBILITY_CHECKLIST.md` — public release reproducibility checklist
- `README_DATA.md` — data organization, cohort, and path requirements
- `RUN_ORDER.txt` — concise execution ordering reference

---

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

If your release package also contains a manuscript-locked configuration template, use it as a reference document rather than as a plug-and-play private environment file.

---

## Recommended execution order

### Path A: final manuscript-aligned training and held-out evaluation

1. Prepare metadata tables and local data directories as described in `README_DATA.md`.
2. Provide the final selected radiomics CSV tables listed under `paths.final_selected_radiomics` in the configuration.
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

3. Lock the chosen hyperparameters or epoch count externally.
4. Execute the final manuscript-aligned training script shown in Path A.

---

## Data requirements and privacy

This repository does **not** include raw CT images, segmentation masks, clinical metadata, or manuscript-private local configuration files. Public users should prepare local data according to `README_DATA.md` and the example metadata structures distributed with the release.

### What is not included

- raw CT images, masks, or clinical metadata
- a filled manuscript-private configuration file with site-specific local paths
- standalone final CLI trainers for the radiomics-only and image-only comparator models

### Data access note

Imaging data, masks, radiomics source materials, and associated metadata may be subject to institutional governance, ethics approval, privacy constraints, or data-use agreements. The public release is therefore limited to workflow code and documentation rather than patient-level study data.

---

## Expected outputs

Depending on the execution path, typical outputs may include:

- final model checkpoints
- held-out internal and external evaluation summaries
- manifest files for cohort and outer-fold definition
- radiomics feature tables and selected-feature subsets
- run logs and release-validation results

The precise output structure depends on the filled configuration and release layout used in your local environment.

---

## Reproducibility notes

This repository is intended for **research transparency, method inspection, and controlled academic reproduction**.

To facilitate responsible reuse:

- review `MANUSCRIPT_ALIGNMENT.md` before mapping repository behavior to manuscript statements
- review `REPRODUCIBILITY_CHECKLIST.md` before public redistribution or third-party reruns
- use configuration templates rather than editing scripts directly
- avoid mixing private institution-specific paths into a public release branch

Although the repository is organized for public reproducibility, successful reruns still depend on environment setup, CUDA compatibility, external dependencies, and locally prepared data that conform to the documented schema.

---

## Environment

Install the required Python packages listed in `requirements.txt`. The `mamba-ssm` backend requires a CUDA-capable environment for actual training and inference with the final fusion model.

A reproducible environment is strongly recommended for publication use. In practice, users should document:

- Python version
- PyTorch and CUDA versions
- MONAI version
- `mamba-ssm` version
- PyRadiomics version
- operating system and GPU information

---

## Intended use

This repository is intended for academic research, reproducibility support, and methodological transparency. It is **not** intended for direct clinical deployment, real-time patient management, or regulatory use.

---

## Citation

If you use this repository in academic work, please cite the associated manuscript once available. If a DOI, preprint link, or journal reference is assigned in the future, it should be added here.

Suggested placeholder format:

```text
Authors. Title. Journal / preprint information. Year.
```

---

## License

This repository is distributed under the MIT License. See the repository-root `LICENSE` file.
