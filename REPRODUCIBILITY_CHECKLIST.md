# Reproducibility Checklist

## Before public release

- [ ] Confirm that all manuscript-reported final numbers come from `training/train_final_multimodal_mamba_fusion.py`.
- [ ] Confirm that held-out internal and external cohorts were not used during optional hyperparameter tuning.
- [ ] Confirm that public scripts no longer contain hard-coded local filesystem defaults.
- [ ] Fill a private copy of `configs/study_config.template.json` with local paths and archive that filled config alongside the exact command lines used for the reported run.
- [ ] Verify that the three final selected radiomics CSV files contain the intended feature set and cohort rows.
- [ ] Verify that the final fixed decision threshold is `0.5` in both manuscript and code.
- [ ] Record the Python, PyTorch, MONAI, scikit-learn, PyRadiomics, and `mamba-ssm` versions used for the reported results.
- [ ] Archive the exact repository snapshot used for submission or revision.
- [ ] Confirm that the repository root contains the final public `LICENSE`.
- [ ] Confirm that all Markdown and text files open cleanly as UTF-8 on another machine.

## Scope checks

- [ ] Ensure the README does not describe architecture-only modules as runnable training scripts.
- [ ] Ensure the manuscript does not claim comparator training drivers that are not included in this public package.
- [ ] Ensure data-access restrictions are stated clearly.
- [ ] Ensure masks are described as mandatory for the released multimodal pipeline.

## Recommended extras

- [ ] Keep example metadata files in sync with the documented schema.
- [ ] Add automated smoke tests for config parsing and file resolution.
- [ ] Add a changelog or release note tying repository tags to manuscript revisions.
