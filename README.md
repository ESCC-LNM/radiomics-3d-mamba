# ESCC LNM Multimodal Prediction Repository

This repository provides the public, manuscript-aligned code release for predicting lymph node metastasis (LNM) in esophageal squamous cell carcinoma (ESCC) from contrast-enhanced CT.

It focuses on a **multimodal framework** that combines:

- **Radiomics features**
- **3D deep-learning image features**

The release is organized for **research transparency, reproducibility, and manuscript-code alignment**.

## Highlights

- Manuscript-aligned multimodal prediction pipeline for ESCC LNM
- Integration of radiomics and volumetric image representations
- Clear separation between preliminary development and final held-out evaluation
- Release-oriented documentation for reproducibility and method inspection

## Repository Scope

This public release includes:

- final multimodal training and evaluation workflow
- preprocessing scripts for cohort handling and radiomics preparation
- model definition files for fusion and comparator architectures
- configuration templates and release-validation utilities

## Repository Structure

- `training/` — training workflows
- `preprocessing/` — data splitting and radiomics preprocessing
- `data/` — dataloading modules
- `models/` — model definitions
- `configs/` — configuration templates
- `utils/` — shared utilities
- `release/` — release validation tools
- `examples/` — example files and references

Key documents:

- `MANUSCRIPT_ALIGNMENT.md`
- `REPRODUCIBILITY_CHECKLIST.md`
- `README_DATA.md`
- `RUN_ORDER.txt`

## Usage

Use the provided configuration template, replace placeholder paths with local settings, prepare data as described in `README_DATA.md`, and run the corresponding training workflow.

## Data Availability

This repository does **not** include raw CT images, segmentation masks, clinical metadata, or private site-specific configuration files.

Users must prepare local data according to the documented format and applicable institutional or ethical requirements.

## Reproducibility

This release is intended for:

- research transparency
- method inspection
- controlled academic reproduction

Reproduction depends on local environment setup, dependencies, hardware, and correctly prepared data.

## Intended Use

This repository is for **academic research only** and is **not intended for clinical deployment**.

## License

Released under the **MIT License**.
