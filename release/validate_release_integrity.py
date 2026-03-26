from pathlib import Path
import json
import sys

PACKAGE_REQUIRED_FILES = [
    "README.md",
    "README_DATA.md",
    "MANUSCRIPT_ALIGNMENT.md",
    "REPRODUCIBILITY_CHECKLIST.md",
    "RUN_ORDER.txt",
    "models/model_multimodal_mamba_fusion.py",
    "training/train_final_multimodal_mamba_fusion.py",
    "data/data_loading_final_multimodal.py",
    "utils/analysis_utils.py",
    "preprocessing/generate_patient_split_manifest.py",
    "preprocessing/extract_radiomics_features.py",
    "preprocessing/select_supervised_radiomics_features.py",
    "release/validate_release_integrity.py",
    "requirements.txt",
    "configs/study_config.template.json",
    "examples/metadata/internal_metadata.example.csv",
    "examples/metadata/external_metadata.example.csv",
]

REPO_ROOT_REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    ".gitignore",
]

UTF8_TEXT_FILES = [
    "README.md",
    "README_DATA.md",
    "MANUSCRIPT_ALIGNMENT.md",
    "REPRODUCIBILITY_CHECKLIST.md",
    "RUN_ORDER.txt",
]


def _fail(message: str) -> None:
    print(message)
    sys.exit(1)


root = Path(__file__).resolve().parents[1]
repo_root = root.parent

missing = [p for p in PACKAGE_REQUIRED_FILES if not (root / p).exists()]
if missing:
    _fail("Missing required package files:\n" + "\n".join(f" - {item}" for item in missing))

missing_root = [p for p in REPO_ROOT_REQUIRED_FILES if not (repo_root / p).exists()]
if missing_root:
    _fail("Missing required repository-root files:\n" + "\n".join(f" - {item}" for item in missing_root))

utf8_failures = []
for rel_path in UTF8_TEXT_FILES:
    try:
        (root / rel_path).read_text(encoding="utf-8")
    except Exception as exc:
        utf8_failures.append(f"{rel_path}: {type(exc).__name__}: {exc}")
try:
    (repo_root / "README.md").read_text(encoding="utf-8")
except Exception as exc:
    utf8_failures.append(f"../README.md: {type(exc).__name__}: {exc}")

if utf8_failures:
    _fail("UTF-8 validation failed:\n" + "\n".join(f" - {item}" for item in utf8_failures))

cfg = json.loads((root / "configs" / "study_config.template.json").read_text(encoding="utf-8"))
expected_top_level = ["paths", "columns", "dataset_roles", "cross_validation", "preprocessing", "runtime", "model", "training"]
missing_top_level = [k for k in expected_top_level if k not in cfg]
if missing_top_level:
    _fail(f"Configuration template missing top-level keys: {missing_top_level}")

expected_path_keys = [
    "internal_metadata_csv",
    "external_metadata_csv",
    "preprocessed_root",
    "radiomics_output_root",
    "selected_features_root",
    "model_selection_output_dir",
    "final_output_dir",
    "images",
    "masks",
    "radiomics",
    "final_selected_radiomics",
]
missing_path_keys = [k for k in expected_path_keys if k not in cfg["paths"]]
if missing_path_keys:
    _fail(f"Configuration template missing path keys: {missing_path_keys}")

for section_name, nested_keys in {
    "images": ["development_dir", "internal_test_dir", "external_test_dir"],
    "masks": ["development_dir", "internal_test_dir", "external_test_dir"],
    "radiomics": ["development_raw_csv", "internal_test_raw_csv", "external_test_raw_csv"],
    "final_selected_radiomics": ["development_csv", "internal_test_csv", "external_test_csv"],
}.items():
    section = cfg["paths"].get(section_name, {})
    missing_nested = [k for k in nested_keys if k not in section]
    if missing_nested:
        _fail(f"Configuration template missing nested keys under paths.{section_name}: {missing_nested}")

print("Release integrity check passed.")
