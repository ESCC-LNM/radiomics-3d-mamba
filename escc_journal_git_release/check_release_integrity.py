from pathlib import Path
import json
import sys

REQUIRED_FILES = [
    'README.md',
    'README_DATA.md',
    'MANUSCRIPT_ALIGNMENT.md',
    'REPRODUCIBILITY_CHECKLIST.md',
    'mamba_fusion_model.py',
    'train_final_mamba.py',
    'data_pipeline_final.py',
    'requirements.txt',
    'configs/study_config.template.json',
]

root = Path(__file__).resolve().parent
missing = [p for p in REQUIRED_FILES if not (root / p).exists()]
if missing:
    print('Missing required release files:')
    for item in missing:
        print(' -', item)
    sys.exit(1)

cfg = json.loads((root / 'configs' / 'study_config.template.json').read_text(encoding='utf-8'))
expected_keys = ['paths', 'columns', 'dataset_roles', 'cross_validation', 'preprocessing', 'runtime', 'model', 'training']
missing_keys = [k for k in expected_keys if k not in cfg]
if missing_keys:
    print('Configuration template missing keys:', missing_keys)
    sys.exit(1)

print('Release integrity check passed.')
