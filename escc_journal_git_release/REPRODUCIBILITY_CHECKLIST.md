# Reproducibility Checklist

## Before public release

- [ ] Confirm that all manuscript-reported final numbers come from the fixed final scripts.
- [ ] Confirm that held-out internal/external cohorts were not used during hyperparameter tuning.
- [ ] Replace all local file-system paths with local configuration or command-line arguments.
- [ ] Replace `LICENSE_PENDING.md` with an approved software license.
- [ ] Archive the exact repository snapshot used for submission.
- [ ] Provide the exact command lines or a manuscript-locked config file.
- [ ] State whether masks are mandatory for all multimodal experiments.
- [ ] Verify that selected radiomics CSV files contain exactly the intended columns.
- [ ] Verify that the final fixed threshold is 0.5 in both manuscript and code.
- [ ] Record the CUDA / PyTorch / MONAI / mamba-ssm environment used for the published results.

## Recommended extras

- [ ] Add a small synthetic example dataset or metadata example.
- [ ] Add automated smoke tests for config parsing and file resolution.
- [ ] Add a changelog that ties repository tags to manuscript revisions.
