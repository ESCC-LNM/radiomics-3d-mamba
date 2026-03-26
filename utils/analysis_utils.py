from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for this operation but is not installed in the current environment."
        )


def configure_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("fusion_journal_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)




def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(v) for v in value.tolist()]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) or math.isinf(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value

def save_json(payload: Mapping[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(payload), f, indent=2, ensure_ascii=False)


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None, *, required: bool = False) -> Any:
    cursor: Any = mapping
    for token in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or token not in cursor:
            if required:
                raise KeyError(f"Missing required configuration key: {dotted_key}")
            return default
        cursor = cursor[token]
    return cursor


def get_required(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    return get_nested(mapping, dotted_key, required=True)


def get_optional(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    return get_nested(mapping, dotted_key, default=default, required=False)


def require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def normalize_sample_id(value: Any) -> str:
    s = str(value).strip()
    for suffix in (".nii.gz", ".nii", ".nii(1).gz"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def normalize_patient_id(value: Any) -> str:
    s = str(value).strip()
    if not s:
        raise ValueError("Encountered an empty patient identifier.")
    return s


def select_device(requested: Optional[str] = None) -> torch.device:
    _require_torch()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if not np.isnan(precision) and not np.isnan(sensitivity) and (precision + sensitivity) > 0 else float("nan")
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "auc": safe_auc(y_true, y_prob),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "brier_score": float(brier),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _threshold_grid(y_prob: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(np.asarray(y_prob, dtype=float), 8))
    anchors = np.array([0.0, 0.5, 1.0], dtype=float)
    grid = np.unique(np.concatenate([unique, anchors]))
    return grid[(grid >= 0.0) & (grid <= 1.0)]


def select_fixed_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    objective: str,
    minimum_specificity: Optional[float] = None,
    minimum_sensitivity: Optional[float] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = _threshold_grid(y_prob)
    rows: List[Dict[str, float]] = []

    for thr in grid:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append(metrics)

    table = pd.DataFrame(rows)

    if minimum_specificity is not None:
        table = table.loc[table["specificity"] >= float(minimum_specificity)].copy()
    if minimum_sensitivity is not None:
        table = table.loc[table["sensitivity"] >= float(minimum_sensitivity)].copy()
    if table.empty:
        table = pd.DataFrame(rows)

    objective = str(objective).lower()
    if objective == "balanced_accuracy":
        score = table["balanced_accuracy"]
    elif objective == "youden_j":
        score = table["sensitivity"] + table["specificity"] - 1.0
    elif objective == "f1":
        score = table["f1"]
    elif objective == "accuracy":
        score = table["accuracy"]
    elif objective == "sensitivity":
        score = table["sensitivity"]
    elif objective == "specificity":
        score = table["specificity"]
    else:
        raise ValueError(
            "Unsupported threshold-selection objective. "
            "Use one of: balanced_accuracy, youden_j, f1, accuracy, sensitivity, specificity."
        )

    table = table.assign(selection_score=np.asarray(score, dtype=float))
    best = table.sort_values(["selection_score", "auc", "threshold"], ascending=[False, False, True]).iloc[0]

    return {
        "fixed_threshold": float(best["threshold"]),
        "objective": objective,
        "minimum_specificity": None if minimum_specificity is None else float(minimum_specificity),
        "minimum_sensitivity": None if minimum_sensitivity is None else float(minimum_sensitivity),
        "selection_score": float(best["selection_score"]),
        "metrics_at_selected_threshold": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v) else None)
            for k, v in best.to_dict().items()
            if k != "selection_score"
        },
    }


def summarise_numeric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def median_epoch_recommendation(best_epochs: Sequence[int]) -> int:
    epochs = [int(v) for v in best_epochs if int(v) > 0]
    if not epochs:
        raise ValueError("No valid best epochs were provided.")
    return int(np.median(np.asarray(epochs, dtype=int)))


def save_feature_list(features: Sequence[str], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        for feat in features:
            f.write(f"{feat}\n")


def resolve_image_path(base_dir: str | Path, sample_id: str, allowed_suffixes: Sequence[str]) -> Path:
    base_dir = Path(base_dir)
    for suffix in allowed_suffixes:
        candidate = base_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve an image file for sample '{sample_id}' inside '{base_dir}'. "
        f"Tried suffixes: {list(allowed_suffixes)}"
    )


def seed_worker(worker_id: int) -> None:
    _require_torch()
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collect_environment_metadata() -> Dict[str, Any]:
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": None,
        "pandas": None,
        "torch": None,
        "sklearn": None,
        "monai": None,
    }
    try:
        versions["numpy"] = np.__version__
    except Exception:
        pass
    try:
        versions["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import monai
        versions["monai"] = monai.__version__
    except Exception:
        pass

    git_commit = None
    git_status_clean = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        git_status_clean = status == ""
    except Exception:
        pass

    return {
        "software_versions": versions,
        "git_commit_hash": git_commit,
        "git_worktree_clean": git_status_clean,
    }


def _bootstrap_sample_by_cluster(cluster_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_clusters = np.asarray(pd.unique(cluster_ids))
    sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    sampled_indices: List[int] = []
    for c in sampled_clusters:
        sampled_indices.extend(np.where(cluster_ids == c)[0].tolist())
    return np.asarray(sampled_indices, dtype=int)


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cluster_ids: Optional[Sequence[Any]] = None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.95,
) -> Dict[str, Dict[str, Optional[float]]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if cluster_ids is None:
        cluster_ids = np.arange(len(y_true))
    cluster_ids = np.asarray(list(cluster_ids))
    rng = np.random.default_rng(seed)

    metrics_to_track = ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "npv", "f1", "brier_score"]
    values: Dict[str, List[float]] = {m: [] for m in metrics_to_track}

    for _ in range(int(n_bootstrap)):
        idx = _bootstrap_sample_by_cluster(cluster_ids, rng)
        y_true_bs = y_true[idx]
        y_prob_bs = y_prob[idx]
        if len(np.unique(y_true_bs)) < 2:
            continue
        metrics = compute_binary_metrics(y_true_bs, y_prob_bs, threshold=threshold)
        for m in metrics_to_track:
            val = metrics[m]
            if not (isinstance(val, float) and math.isnan(val)):
                values[m].append(float(val))

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name, arr in values.items():
        if len(arr) == 0:
            out[metric_name] = {"lower": None, "upper": None}
            continue
        out[metric_name] = {
            "lower": float(np.quantile(arr, lower_q)),
            "upper": float(np.quantile(arr, upper_q)),
        }
    return out
