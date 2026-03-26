# ESCC Journal Release

This repository root wraps the manuscript-aligned ESCC release package stored in `escc/`.

## Where to start

- Main project documentation: `escc/README.md`
- Data schema notes: `escc/README_DATA.md`
- Release checklist: `escc/REPRODUCIBILITY_CHECKLIST.md`

## Repository layout

- `escc/`: runnable scripts, model definitions, config template, and example metadata
- `LICENSE`: repository-wide software license
- `.gitignore`: repository-wide ignore rules for caches, logs, checkpoints, and model artefacts

Run code from inside `escc/`, for example:

```bash
cd escc
python -m training.train_final_multimodal_mamba_fusion --config configs/study_config.template.json
```
