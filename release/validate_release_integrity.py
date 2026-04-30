from pathlib import Path
import json
import sys

PACKAGE_REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "README_DATA.md",
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
    "examples/metadata/external_test1_metadata.example.csv",
    "examples/metadata/external_test2_metadata.example.csv",
]

UTF8_TEXT_FILES = [
    "README.md",
    "README_DATA.md",
    "REPRODUCIBILITY_CHECKLIST.md",
    "RUN_ORDER.txt",
]


def _fail(message: str) -> None:
    print(message)
    sys.exit(1)


root = Path(__file__).resolve().parents[1]

missing = [p for p in PACKAGE_REQUIRED_FILES if not (root / p).exists()]
if missing:
    _fail("Missing required package files:\n" + "\n".join(f" - {item}" for item in missing))

utf8_failures = []
for rel_path in UTF8_TEXT_FILES:
    try:
        (root / rel_path).read_text(encoding="utf-8")
    except Exception as exc:
        utf8_failures.append(f"{rel_path}: {type(exc).__name__}: {exc}")
if utf8_failures:
    _fail("UTF-8 validation failed:\n" + "\n".join(f" - {item}" for item in utf8_failures))

cfg = json.loads((root / "configs" / "study_config.template.json").read_text(encoding="utf-8-sig"))
expected_top_level = [
    "paths",
    "columns",
    "dataset_roles",
    "dataset",
    "cross_validation",
    "feature_selection",
    "preprocessing",
    "runtime",
    "model",
    "training",
]
missing_top_level = [k for k in expected_top_level if k not in cfg]
if missing_top_level:
    _fail(f"Configuration template missing top-level keys: {missing_top_level}")

expected_path_keys = [
    "internal_metadata_csv",
    "external_test1_metadata_csv",
    "external_test2_metadata_csv",
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

expected_column_keys = ["sample_id", "patient_id", "label", "group", "external_patient_id"]
missing_column_keys = [k for k in expected_column_keys if k not in cfg["columns"]]
if missing_column_keys:
    _fail(f"Configuration template missing column keys: {missing_column_keys}")

expected_role_keys = [
    "development_group_value",
    "internal_test_group_value",
    "external_test1_role_name",
    "external_test2_role_name",
]
missing_role_keys = [k for k in expected_role_keys if k not in cfg["dataset_roles"]]
if missing_role_keys:
    _fail(f"Configuration template missing dataset role keys: {missing_role_keys}")

for section_name, nested_keys in {
    "images": ["development_dir", "internal_test_dir", "external_test1_dir", "external_test2_dir"],
    "masks": ["development_dir", "internal_test_dir", "external_test1_dir", "external_test2_dir"],
    "radiomics": ["development_raw_csv", "internal_test_raw_csv", "external_test1_raw_csv", "external_test2_raw_csv"],
    "final_selected_radiomics": ["development_csv", "internal_test_csv", "external_test1_csv", "external_test2_csv"],
}.items():
    section = cfg["paths"].get(section_name, {})
    missing_nested = [k for k in nested_keys if k not in section]
    if missing_nested:
        _fail(f"Configuration template missing nested keys under paths.{section_name}: {missing_nested}")

print("Release integrity check passed.")
