"""
 Mamba + Radiomics Fusion Trainer (5-Fold CV)

GitHub-ready training entry:
- GitHub-friendly defaults (relative ./data paths).
- Privacy-first: IDs are NOT returned by default; avoid printing paths/IDs in logs.
- Anti-leakage: Radiomics preprocessing is fold-isolated in data_pipeline_fusion.py.

Notes:
- This script assumes `get_fusion_dataloaders(args)` comes from data_pipeline_fusion.py
  and handles fold split + fold-isolated impute/scale internally.
"""

from __future__ import annotations

import sys
import time
import argparse
import logging
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


# =============================================================================
# Project imports (robust for both "python train_fusion_mamba.py" and package usage)
# =============================================================================
def _setup_imports():
    try:
        # 1) If executed as a module inside a package: "python -m yourpkg.train_fusion_mamba"
        try:
            if __package__:
                from .data_pipeline_fusion import get_fusion_dataloaders  # type: ignore
                from .mamba_fusion_model import Mamba_Fusion_Model  # type: ignore
                return get_fusion_dataloaders, Mamba_Fusion_Model
        except Exception:
            pass

        # 2) If executed as a plain script: "python train_fusion_mamba.py"
        current_file_path = Path(__file__).resolve()
        mod_dir = current_file_path.parent
        if str(mod_dir) not in sys.path:
            sys.path.insert(0, str(mod_dir))

        from data_pipeline_fusion import get_fusion_dataloaders  # noqa
        from mamba_fusion_model import Mamba_Fusion_Model  # noqa

        return get_fusion_dataloaders, Mamba_Fusion_Model

    except Exception as e:
        print("=" * 80)
        print("!!! CRITICAL IMPORT ERROR in train_fusion_mamba.py !!!")
        print(e)
        print("=" * 80)
        raise


get_fusion_dataloaders, Mamba_Fusion_Model = _setup_imports()

# =============================================================================
# Logging
# =============================================================================
LOG_DIR = Path("./outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILENAME = LOG_DIR / "train_fusion_mamba.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("MambaFusion")

# =============================================================================
# Final locked config for paper release (single source of truth)
# =============================================================================
FINAL_CFG: Dict[str, Any] = {
    # training
    "lr": 1e-4,
    "weight_decay": 3e-5,
    "batch_size": 16,
    "epochs": 220,
    "n_splits": 5,
    "num_workers": 4,

    # threshold calibration
    "min_spec": 0.70,

    # scheduler & early stop
    "use_cosine": True,
    "warmup_epochs": 10,
    "early_stop": True,
    "patience": 30,

    # data / CT window
    "roi_size": [32, 64, 64],
    "window_width": 400.0,
    "window_level": 40.0,

    # model
    "mamba_embed_dim": 128,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "rad_input_features": 112,
    "rad_hidden_features": 64,
    "fusion_hidden_features": 64,
    "num_classes": 2,
    "dropout_rate": 0.3,

    # fusion scaling
    "rad_scale": 0.3,

    # aux heads
    "aux_img_weight": 0.1,
    "aux_rad_weight": 0.05,

    # switches
    "use_cache": True,
    "disable_aux": False,
    "disable_gate": False,

    # reproducibility & pipeline knobs
    "seed": 42,
    "nifti_exts": [".nii.gz", ".nii", ".nii(1).gz"],

    # privacy (default OFF)
    "return_ids": False,
    "anonymize_ids": False,
}


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        from monai.utils import set_determinism  # type: ignore
        set_determinism(seed=seed)
    except Exception:
        pass
    log.info(f"--- [Global Seed] Set to {seed} for reproducibility ---")


# =============================================================================
# CLI args
# =============================================================================
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mamba-Radiomics Fusion Trainer (5-Fold CV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Mode ----
    g_mode = parser.add_argument_group("Mode")
    g_mode.add_argument(
        "--unlocked",
        action="store_true",
        help="Do NOT override CLI args with FINAL_CFG (for ablations/experiments).",
    )

    # ---- Trainer
    g_train = parser.add_argument_group("Trainer")
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--weight_decay", type=float, default=1e-5)
    g_train.add_argument("--batch_size", type=int, default=16)
    g_train.add_argument("--epochs", type=int, default=200)
    g_train.add_argument("--n_splits", type=int, default=5)
    g_train.add_argument("--num_workers", type=int, default=4)
    g_train.add_argument("--min_spec", type=float, default=0.70)
    g_train.add_argument("--use_cosine", action="store_true")
    g_train.add_argument("--warmup_epochs", type=int, default=10)
    g_train.add_argument("--early_stop", action="store_true")
    g_train.add_argument("--patience", type=int, default=25)

    g_train.add_argument(
        "--eval_tests_during_train",
        action="store_true",
        help="If set, evaluate internal/external test sets whenever a new best validation checkpoint is found. "
             "Recommended to keep OFF to preserve strict test-set isolation.",
    )

    g_train.add_argument(
        "--run_fold",
        type=int,
        default=None,
        help="If set, only run this (1-based) fold.",
    )
    g_train.add_argument("--seed", type=int, default=42)

    g_train.add_argument(
        "--no_cache",
        dest="use_cache",
        action="store_false",
        help="Disable MONAI CacheDataset.",
    )
    parser.set_defaults(use_cache=True)

    # ---- Paths (matching README structure) ----
    g_path = parser.add_argument_group("Paths")

    g_path.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/checkpoints"),
        help="Directory to save models and CSVs.",
    )

    base_dir = Path("./data")
    rad_dir = Path("./outputs/selected_features/global")

    # Optional: root dir for fold-specific selected radiomics (expects fold01..fold0K subfolders)
    # If set, the trainer will automatically pick foldXX CSVs per fold.
    # Example: --rad_root_dir ./outputs/selected_features
    g_path.add_argument(
        "--rad_root_dir",
        type=Path,
        default=None,
        help="Root folder containing fold01..fold0K selected radiomics. If set, overrides --rad_*_csv per fold.",
    )


    g_path.add_argument("--internal_csv", type=Path, default=base_dir / "internal_group.csv")
    g_path.add_argument("--external_csv", type=Path, default=base_dir / "external_label.csv")

    g_path.add_argument("--trainval_img_dir", type=Path, default=base_dir / "trainval_images")
    g_path.add_argument("--internal_test_img_dir", type=Path, default=base_dir / "internal_test_images")
    g_path.add_argument("--external_test_img_dir", type=Path, default=base_dir / "external_test_images")

    g_path.add_argument("--rad_trainval_csv", type=Path, default=rad_dir / "radiomics_internal_trainval_sel.csv")
    g_path.add_argument("--rad_internal_test_csv", type=Path, default=rad_dir / "radiomics_internal_test_sel.csv")
    g_path.add_argument("--rad_external_test_csv", type=Path, default=rad_dir / "radiomics_external_test_sel.csv")

    # ---- Data / MONAI
    g_data = parser.add_argument_group("Data & MONAI")
    g_data.add_argument("--roi_size", type=int, nargs=3, default=[32, 64, 64])
    g_data.add_argument("--window_width", type=float, default=400.0)
    g_data.add_argument("--window_level", type=float, default=40.0)

    # ---- Pipeline I/O
    g_io = parser.add_argument_group("Pipeline I/O")
    g_io.add_argument(
        "--nifti_exts",
        type=str,
        nargs="+",
        default=[".nii.gz", ".nii", ".nii(1).gz"],
        help="Candidate NIfTI extensions for resolving file by ID.",
    )

    # ---- Model
    g_model = parser.add_argument_group("Model")
    g_model.add_argument("--mamba_embed_dim", type=int, default=128)
    g_model.add_argument("--mamba_d_state", type=int, default=16)
    g_model.add_argument("--mamba_d_conv", type=int, default=4)
    g_model.add_argument("--mamba_expand", type=int, default=2)
    g_model.add_argument("--rad_input_features", type=int, default=112)
    g_model.add_argument("--rad_hidden_features", type=int, default=64)
    g_model.add_argument("--fusion_hidden_features", type=int, default=64)
    g_model.add_argument("--num_classes", type=int, default=2)
    g_model.add_argument("--dropout_rate", type=float, default=0.3)
    g_model.add_argument("--rad_scale", type=float, default=0.3)

    # ---- Aux / gating
    g_fuse = parser.add_argument_group("Aux & Gating")
    g_fuse.add_argument("--aux_img_weight", type=float, default=0.1)
    g_fuse.add_argument("--aux_rad_weight", type=float, default=0.05)
    g_fuse.add_argument("--disable_aux", action="store_true")
    g_fuse.add_argument("--disable_gate", action="store_true")

    # ---- Privacy
    g_priv = parser.add_argument_group("Privacy")
    g_priv.add_argument("--return_ids", action="store_true", help="Return sample IDs from pipeline (default off).")
    g_priv.add_argument("--anonymize_ids", action="store_true", help="Hash IDs if returning them (default off).")

    return parser.parse_args()


# =============================================================================
# Metrics
# =============================================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc = 0.5

    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    sens, spec = 0.0, 0.0
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        pass

    return {
        "AUC": float(auc),
        "ACC": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Threshold": float(threshold),
    }


def compute_class_weights_from_loader(loader, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        labels = batch["label"].detach().cpu().numpy().astype(int)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "balance_with_floor",
    min_spec: float = 0.70,
) -> float:
    """
    strategy="balance_with_floor":
      Among thresholds with Spec >= min_spec, pick the one minimizing |Sens - Spec|.
      Fallback to Youden's J if none satisfy the floor.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
    except ValueError:
        return 0.5

    spec = 1.0 - fpr
    sens = tpr

    if strategy == "balance_with_floor" and min_spec is not None:
        ok = spec >= float(min_spec)
        if np.any(ok):
            diff = np.abs(sens[ok] - spec[ok])
            idx_rel = int(np.argmin(diff))
            return float(thr[ok][idx_rel])

    youden = sens - fpr
    idx = int(np.argmax(youden))
    return float(thr[idx])


def compute_inference_time_per_case(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            rad_features = batch["rad_features"].to(device)
            bs = int(images.size(0))

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            _ = model({"image": images, "rad_features": rad_features})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - start

            total_time += elapsed
            total_samples += bs

    return float(total_time / total_samples) if total_samples > 0 else 0.0


# =============================================================================
# Train / Validate (Float32; AMP disabled for reproducibility)
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: Adam,
    criterion_main,
    criterion_aux,
    device: torch.device,
    use_aux: bool,
    aux_img_weight: float,
    aux_rad_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    max_grad_norm = 10.0

    for batch_data in loader:
        images = batch_data["image"].to(device)
        rad_features = batch_data["rad_features"].to(device)
        labels = batch_data["label"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model({"image": images, "rad_features": rad_features})

        # Compatibility: some models may return (logits, u, gate)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if use_aux and isinstance(outputs, dict):
            logits_fusion = outputs["fusion"]
            logits_img = outputs.get("img", None)
            logits_rad = outputs.get("rad", None)

            loss = criterion_main(logits_fusion, labels)
            if logits_img is not None:
                loss = loss + float(aux_img_weight) * criterion_aux(logits_img, labels)
            if logits_rad is not None:
                loss = loss + float(aux_rad_weight) * criterion_aux(logits_rad, labels)
        else:
            logits_fusion = outputs["fusion"] if isinstance(outputs, dict) else outputs
            loss = criterion_main(logits_fusion, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def validate(
    model: nn.Module,
    loader,
    criterion_val,
    device: torch.device,
    threshold: float = 0.5,
    return_raw: bool = False,
):
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device)
            rad_features = batch_data["rad_features"].to(device)
            labels = batch_data["label"].to(device).long()

            logits = model({"image": images, "rad_features": rad_features})

            # Compatibility: eval may return tuple (logits, u, gate) or dict
            if isinstance(logits, tuple):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = logits["fusion"]

            loss = criterion_val(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += float(loss.item())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    y_true = np.array(all_labels, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    if return_raw:
        return avg_loss, y_true, y_prob

    metrics = calculate_metrics(y_true, y_prob, threshold=float(threshold))
    return avg_loss, metrics


# =============================================================================
# Safety checks (make common GitHub-user mistakes fail fast with clear messages)
# =============================================================================
def _peek_radiomics_feature_dim(rad_csv: Path) -> Optional[int]:
    """
    Count feature columns of selected radiomics CSV.
    Expected format from select_radiomics_features.py: ID + feature columns.
    """
    if not rad_csv.exists():
        return None
    try:
        df_head = pd.read_csv(rad_csv, nrows=1)
    except Exception:
        return None

    cols = list(df_head.columns)
    # Prefer exact "ID", otherwise treat first column as ID (aligns with data_pipeline_fusion._read_radiomics_csv)
    if "ID" in cols:
        feat_cols = [c for c in cols if c != "ID"]
    else:
        feat_cols = cols[1:]

    # Filter out obviously-non-feature columns defensively
    drop_like = {"label", "y", "target", "group", "split", "fold"}
    feat_cols = [c for c in feat_cols if str(c).strip().lower() not in drop_like]
    return int(len(feat_cols))


def assert_radiomics_dim_or_die(args: argparse.Namespace) -> None:
    """
    If selected feature count != args.rad_input_features, raise a clear error.
    This avoids silent shape mismatch later.
    """
    dim = _peek_radiomics_feature_dim(Path(args.rad_trainval_csv))
    if dim is None:
        log.warning("[Check] Radiomics CSV not found yet or unreadable: %s", args.rad_trainval_csv)
        return

    if int(dim) != int(args.rad_input_features):
        msg = (
            f"Radiomics feature dim mismatch!\n"
            f"  - CSV: {args.rad_trainval_csv} has {dim} feature columns (excluding ID)\n"
            f"  - Config expects rad_input_features={args.rad_input_features}\n\n"
            f"Fix:\n"
            f"  (A) Regenerate selected radiomics with top_k={args.rad_input_features}.\n"
            f"      Example: python select_radiomics_features.py --mode global --top_k {args.rad_input_features}\n"
            f"  (B) Or run this trainer with --unlocked and set --rad_input_features {dim}.\n"
        )
        raise ValueError(msg)


def apply_fold_radiomics_paths(args: argparse.Namespace, fold_id_1based: int) -> None:
    """If args.rad_root_dir is set, override radiomics CSV paths for the given fold."""
    if getattr(args, "rad_root_dir", None) is None:
        return
    root: Path = Path(args.rad_root_dir)
    fold_dir = root / f"fold{fold_id_1based:02d}"
    if not fold_dir.exists():
        raise FileNotFoundError(
            f"--rad_root_dir was set to '{root}', but expected fold directory not found: {fold_dir}"
        )
    args.rad_trainval_csv = fold_dir / "radiomics_internal_trainval_sel.csv"
    args.rad_internal_test_csv = fold_dir / "radiomics_internal_test_sel.csv"
    args.rad_external_test_csv = fold_dir / "radiomics_external_test_sel.csv"


# =============================================================================
# Config override
# =============================================================================
def override_args_with_final_cfg(args: argparse.Namespace) -> argparse.Namespace:
    """
    Override key hyperparameters for paper reproducibility.
    Intentionally does NOT override file paths.
    """
    # training
    args.lr = FINAL_CFG["lr"]
    args.weight_decay = FINAL_CFG["weight_decay"]
    args.batch_size = FINAL_CFG["batch_size"]
    args.epochs = FINAL_CFG["epochs"]
    args.n_splits = FINAL_CFG["n_splits"]
    args.num_workers = FINAL_CFG["num_workers"]

    # threshold
    args.min_spec = FINAL_CFG["min_spec"]

    # scheduler / early stop
    args.use_cosine = FINAL_CFG["use_cosine"]
    args.warmup_epochs = FINAL_CFG["warmup_epochs"]
    args.early_stop = FINAL_CFG["early_stop"]
    args.patience = FINAL_CFG["patience"]

    # data / window
    args.roi_size = FINAL_CFG["roi_size"]
    args.window_width = FINAL_CFG["window_width"]
    args.window_level = FINAL_CFG["window_level"]

    # model
    args.mamba_embed_dim = FINAL_CFG["mamba_embed_dim"]
    args.mamba_d_state = FINAL_CFG["mamba_d_state"]
    args.mamba_d_conv = FINAL_CFG["mamba_d_conv"]
    args.mamba_expand = FINAL_CFG["mamba_expand"]
    args.rad_input_features = FINAL_CFG["rad_input_features"]
    args.rad_hidden_features = FINAL_CFG["rad_hidden_features"]
    args.fusion_hidden_features = FINAL_CFG["fusion_hidden_features"]
    args.num_classes = FINAL_CFG["num_classes"]
    args.dropout_rate = FINAL_CFG["dropout_rate"]
    args.rad_scale = FINAL_CFG["rad_scale"]

    # aux / switches
    args.aux_img_weight = FINAL_CFG["aux_img_weight"]
    args.aux_rad_weight = FINAL_CFG["aux_rad_weight"]
    args.use_cache = FINAL_CFG["use_cache"]
    args.disable_aux = FINAL_CFG["disable_aux"]
    args.disable_gate = FINAL_CFG["disable_gate"]

    # pipeline + privacy
    args.seed = FINAL_CFG["seed"]
    args.nifti_exts = list(FINAL_CFG["nifti_exts"])
    args.return_ids = bool(FINAL_CFG["return_ids"])
    args.anonymize_ids = bool(FINAL_CFG["anonymize_ids"])

    return args


# =============================================================================
# Main
# =============================================================================
def main(args: argparse.Namespace) -> None:
    # Paper mode by default; opt-out with --unlocked
    if not bool(getattr(args, "unlocked", False)):
        args = override_args_with_final_cfg(args)

    set_seed(int(getattr(args, "seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fail fast for the most common public-repo pitfall: radiomics dim mismatch
    # If --rad_root_dir is used (fold-specific), we validate per-fold inside the loop.
    if getattr(args, "rad_root_dir", None) is None:
        assert_radiomics_dim_or_die(args)

    log.info("=====================================================================")
    log.info("Mamba-Radiomics Fusion Training (5-Fold CV)")
    log.info("=====================================================================")
    log.info(f"Using device        : {device}")
    log.info(f"Output directory    : {args.output_dir}")
    log.info(f"Input Radiomics Dir : {Path(args.rad_trainval_csv).parent}")
    if getattr(args, "rad_root_dir", None) is not None:
        log.info(f"Fold Radiomics Root : {Path(args.rad_root_dir)}")
    log.info(f"ROI size            : {tuple(args.roi_size)}")
    log.info(f"Batch size          : {args.batch_size}")
    log.info(f"Learning rate       : {args.lr}")
    log.info(f"Weight Decay (L2)   : {args.weight_decay}")
    log.info(f"Use CacheDataset    : {args.use_cache}")
    log.info(f"Min Spec (ThrCalib) : {args.min_spec} (balance_with_floor)")
    log.info("Precision           : Float32 (AMP disabled)")
    log.info(f"Use Cosine LR       : {args.use_cosine}")
    log.info(f"Warmup Epochs       : {args.warmup_epochs}")
    log.info(f"Early Stop          : {args.early_stop} (patience={args.patience})")
    log.info(f"Mamba embed/state   : dim={args.mamba_embed_dim}, d_state={args.mamba_d_state}")
    log.info(f"Radiomics feat dim  : input={args.rad_input_features}, hidden={args.rad_hidden_features}")
    log.info(f"Fusion hidden dim   : {args.fusion_hidden_features}")
    log.info(f"Radiomics scale     : {args.rad_scale}")
    log.info("Privacy             : return_ids=%s, anonymize_ids=%s", args.return_ids, args.anonymize_ids)
    log.info("Config mode         : %s", "UNLOCKED (respect CLI)" if args.unlocked else "LOCKED (FINAL_CFG)")
    log.info("")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_internal_metrics: List[Dict[str, float]] = []
    all_external_metrics: List[Dict[str, float]] = []
    all_inference_times: List[float] = []

    if getattr(args, "run_fold", None) is not None:
        folds_to_run = [int(args.run_fold) - 1]
        log.warning(f"--- Running ONLY Fold {args.run_fold} ---")
    else:
        folds_to_run = list(range(int(args.n_splits)))
        log.info(f"--- Starting {args.n_splits}-Fold Cross-Validation ---")

    for fold_idx in folds_to_run:
        fold_id = fold_idx + 1
        log.info("")
        log.info(f"------------------------------ Fold {fold_id}/{args.n_splits} ------------------------------")
        args.fold_idx = fold_idx

        # Resolve fold-specific selected radiomics paths (if --rad_root_dir is set)
        apply_fold_radiomics_paths(args, fold_id)

        # Fail fast: radiomics dim mismatch (checked per fold when using --rad_root_dir)
        assert_radiomics_dim_or_die(args)

        # 1) DataLoaders
        log.info("  [Data] Building fusion dataloaders ...")
        dataloaders = get_fusion_dataloaders(args)
        log.info(
            "  [Data] Done. "
            f"train={len(dataloaders['train'])}, "
            f"val={len(dataloaders['val'])}, "
            f"internal_test={len(dataloaders['internal_test'])}, "
            f"external={len(dataloaders['external'])}"
        )

        # 2) class weights (from train only)
        class_weights = compute_class_weights_from_loader(
            dataloaders["train"],
            num_classes=int(args.num_classes),
        ).to(device)
        log.info(f"  [Class Weights] {class_weights.detach().cpu().numpy().tolist()}")

        # 3) model
        model = Mamba_Fusion_Model(
            mamba_embed_dim=int(args.mamba_embed_dim),
            mamba_d_state=int(args.mamba_d_state),
            mamba_d_conv=int(args.mamba_d_conv),
            mamba_expand=int(args.mamba_expand),
            rad_input_features=int(args.rad_input_features),
            rad_hidden_features=int(args.rad_hidden_features),
            fusion_hidden_features=int(args.fusion_hidden_features),
            num_classes=int(args.num_classes),
            dropout_rate=float(args.dropout_rate),
            use_aux_cls=not bool(args.disable_aux),
            aux_img_weight=float(args.aux_img_weight),
            aux_rad_weight=float(args.aux_rad_weight),
            use_gated_fusion=not bool(args.disable_gate),
            rad_scale=float(args.rad_scale),
        ).to(device)

        # 4) loss / optimizer / scheduler
        criterion_main = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        criterion_aux = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        criterion_val = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = Adam(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        scheduler = None
        if bool(args.use_cosine):
            from torch.optim.lr_scheduler import CosineAnnealingLR

            t_max = max(1, int(args.epochs) - int(args.warmup_epochs))
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
            log.info(
                f"  [Scheduler] CosineAnnealingLR enabled "
                f"with T_max={t_max}, warmup_epochs={args.warmup_epochs}"
            )

        use_aux_flag = not bool(args.disable_aux)

        best_val_auc = 0.0
        best_epoch = 0
        best_threshold = 0.5
        best_internal = None
        best_external = None
        no_improve_epochs = 0

        model_save_path = args.output_dir / f"mamba_fusion_fold{fold_id}_best.pth"

        # 5) train loop
        for epoch in range(1, int(args.epochs) + 1):
            log.info(f"--- Epoch {epoch:03d}/{args.epochs} (Fold {fold_id}) ---")

            train_loss = train_one_epoch(
                model=model,
                loader=dataloaders["train"],
                optimizer=optimizer,
                criterion_main=criterion_main,
                criterion_aux=criterion_aux,
                device=device,
                use_aux=use_aux_flag,
                aux_img_weight=float(args.aux_img_weight),
                aux_rad_weight=float(args.aux_rad_weight),
            )

            # warmup + cosine
            if bool(args.use_cosine) and scheduler is not None:
                if epoch <= int(args.warmup_epochs):
                    warmup_lr = float(args.lr) * epoch / max(1, int(args.warmup_epochs))
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                log.info(f"  Current LR        : {current_lr:.8f}")

            if np.isnan(train_loss):
                log.error(f"  !!! Train Loss is NaN. Stopping Fold {fold_id}. !!!")
                break

            log.info(f"  Train Loss        : {train_loss:.4f}")

            # ---- Val (threshold=0.5 for monitoring) ----
            val_loss, val_metrics = validate(
                model=model,
                loader=dataloaders["val"],
                criterion_val=criterion_val,
                device=device,
                threshold=0.5,
            )

            log.info(
                "  Val   Loss        : "
                f"{val_loss:.4f} | "
                f"AUC={val_metrics['AUC']:.4f} | "
                f"ACC={val_metrics['ACC']:.4f} | "
                f"Sens={val_metrics['Sensitivity']:.4f} | "
                f"Spec={val_metrics['Specificity']:.4f}"
            )

            # ---- Best model selection (by Val AUC only) ----
            if val_metrics["AUC"] > best_val_auc:
                best_val_auc = float(val_metrics["AUC"])
                best_epoch = int(epoch)
                no_improve_epochs = 0

                # Calibrate threshold using Val only (Spec floor)
                _, v_labels, v_probs = validate(
                    model=model,
                    loader=dataloaders["val"],
                    criterion_val=criterion_val,
                    device=device,
                    return_raw=True,
                )
                best_threshold = find_optimal_threshold(
                    v_labels,
                    v_probs,
                    strategy="balance_with_floor",
                    min_spec=float(args.min_spec),
                )

                torch.save(model.state_dict(), model_save_path)
                log.info(
                    f"  >> New Best Val AUC: {best_val_auc:.4f} | "
                    f"Calibrated Thr (balance_with_floor): {best_threshold:.4f}"
                )
                log.info(f"  >> Saved best model: {model_save_path}")

                # (Optional) Evaluate held-out test sets during training (NOT recommended)
                if bool(args.eval_tests_during_train):
                    _, best_internal = validate(
                        model=model,
                        loader=dataloaders["internal_test"],
                        criterion_val=criterion_val,
                        device=device,
                        threshold=float(best_threshold),
                    )
                    _, best_external = validate(
                        model=model,
                        loader=dataloaders["external"],
                        criterion_val=criterion_val,
                        device=device,
                        threshold=float(best_threshold),
                    )
                    best_internal["Threshold"] = float(best_threshold)
                    best_external["Threshold"] = float(best_threshold)


            else:
                no_improve_epochs += 1
                if bool(args.early_stop) and no_improve_epochs >= int(args.patience):
                    log.info(
                        f"  [Early Stop] No Val AUC improvement for "
                        f"{no_improve_epochs} epochs (patience={args.patience}). "
                        f"Stopping training for Fold {fold_id}."
                    )
                    break

        
        # ---------------------------------------------------------------------
        # Strict held-out evaluation (recommended): evaluate internal/external ONCE
        # after training ends, using the best checkpoint chosen by validation AUC.
        # ---------------------------------------------------------------------
        if (best_internal is None) or (best_external is None):
            if model_save_path.exists():
                try:
                    state = torch.load(model_save_path, map_location=device)
                    model.load_state_dict(state)
                    model.eval()
                    _, best_internal = validate(
                        model=model,
                        loader=dataloaders["internal_test"],
                        criterion_val=criterion_val,
                        device=device,
                        threshold=float(best_threshold),
                    )
                    _, best_external = validate(
                        model=model,
                        loader=dataloaders["external"],
                        criterion_val=criterion_val,
                        device=device,
                        threshold=float(best_threshold),
                    )
                    best_internal["Threshold"] = float(best_threshold)
                    best_external["Threshold"] = float(best_threshold)
                except Exception as e:
                    log.warning(f"  !!! WARNING: Failed strict held-out evaluation for Fold {fold_id}: {e}")
            else:
                log.warning(f"  !!! WARNING: Best checkpoint not found for Fold {fold_id}: {model_save_path}")
        log.info(
            f"--- Fold {fold_id} Complete | "
            f"Best Epoch: {best_epoch:03d} | Best Val AUC: {best_val_auc:.4f} ---"
        )

        if best_internal is not None and best_external is not None:
            log.info(
                "  [Final Internal-Test @BestModel] "
                f"AUC={best_internal['AUC']:.4f}, "
                f"ACC={best_internal['ACC']:.4f}, "
                f"Sens={best_internal['Sensitivity']:.4f}, "
                f"Spec={best_internal['Specificity']:.4f}, "
                f"Thr={best_internal['Threshold']:.4f}"
            )
            log.info(
                "  [Final External-Test @BestModel] "
                f"AUC={best_external['AUC']:.4f}, "
                f"ACC={best_external['ACC']:.4f}, "
                f"Sens={best_external['Sensitivity']:.4f}, "
                f"Spec={best_external['Specificity']:.4f}, "
                f"Thr={best_external['Threshold']:.4f}"
            )
            all_internal_metrics.append(best_internal)
            all_external_metrics.append(best_external)
        else:
            log.warning("  !!! WARNING: No valid best metrics recorded for this fold !!!")

        # Timing on external set (using best weights)
        if model_save_path.exists():
            best_model = Mamba_Fusion_Model(
                mamba_embed_dim=int(args.mamba_embed_dim),
                mamba_d_state=int(args.mamba_d_state),
                mamba_d_conv=int(args.mamba_d_conv),
                mamba_expand=int(args.mamba_expand),
                rad_input_features=int(args.rad_input_features),
                rad_hidden_features=int(args.rad_hidden_features),
                fusion_hidden_features=int(args.fusion_hidden_features),
                num_classes=int(args.num_classes),
                dropout_rate=float(args.dropout_rate),
                use_aux_cls=not bool(args.disable_aux),
                aux_img_weight=float(args.aux_img_weight),
                aux_rad_weight=float(args.aux_rad_weight),
                use_gated_fusion=not bool(args.disable_gate),
                rad_scale=float(args.rad_scale),
            ).to(device)
            state_dict = torch.load(model_save_path, map_location=device)
            best_model.load_state_dict(state_dict)

            inf_time = compute_inference_time_per_case(best_model, dataloaders["external"], device)
            all_inference_times.append(float(inf_time))
            log.info(
                f"  [Timing] Fold {fold_id}: "
                f"Avg inference time per case on External = {inf_time:.4f} s"
            )
        else:
            log.warning("  [Timing] Best model file not found; skip timing.")

    # =============================================================================
    # Summary
    # =============================================================================
    log.info("")
    log.info("============================== Cross-Validation Summary ==============================")

    def summarize(name: str, metrics_list: List[Dict[str, float]]) -> None:
        if not metrics_list:
            log.info(f"--- {name}: No results ---")
            return
        df = pd.DataFrame(metrics_list)
        csv_path = args.output_dir / f"final_metrics_{name.lower().replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Saved {name} metrics to CSV: {csv_path}")
        for metric in ["AUC", "ACC", "Sensitivity", "Specificity"]:
            if metric in df.columns:
                vals = df[metric].astype(float).values
                log.info(f"  Mean {metric:<12}: {np.mean(vals):.4f} (Std: {np.std(vals):.4f})")

    if getattr(args, "run_fold", None) is None:
        summarize("Internal Test Set", all_internal_metrics)
        summarize("External Test Set", all_external_metrics)

        if all_inference_times:
            mean_t = float(np.mean(all_inference_times))
            std_t = float(np.std(all_inference_times))
            log.info(
                f"[Timing] External Test - Avg inference time per case across folds: "
                f"{mean_t:.4f} s (Std: {std_t:.4f})"
            )
    else:
        log.info("Single-fold mode: skip aggregated summary.")

    log.info("--- Mamba-Radiomics Fusion Training script finished ---")


# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    args_init = get_args()

    if not torch.cuda.is_available():
        log.error("=" * 80)
        log.error("!!! WARNING: CUDA is not available, running on CPU !!!")
        log.error("3D model training on CPU will be extremely slow.")
        log.error("=" * 80)

    try:
        main(args_init)
    except Exception as e:
        log.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)
