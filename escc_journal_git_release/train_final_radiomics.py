"""
Final Global Training Script for Radiomics-Only Baseline (Optimized 2-Layer MLP)

Must match train_final_mamba.py:
1) Fixed-Epoch Training: hard stop at Epoch 220 (no validation, no early stopping).
2) Prospective Evaluation: uses the same fixed global threshold (0.5).
3) Test Set Isolation: evaluated ONLY once post-training.
4) Same split logic and preprocessing protocol for fair comparison.

IMPORTANT:
- Uses radiomics-only dataloaders. Batches contain: rad_features + label + id.
- Fair cohort policy defaults to image+radiomics intersection in data_pipeline_final.py.
"""

import os
import sys
import time
import argparse
import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

try:
    from data_pipeline_final import get_final_radiomics_dataloaders
except ImportError as e:
    sys.stderr.write(f"Import Error: {e}\nEnsure data_pipeline_final.py is in scope.\n")
    sys.exit(1)

# =============================================================================
# Logging Setup
# =============================================================================
LOG_FILENAME = "train_final_radiomics.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILENAME, mode="w")],
)
log = logging.getLogger("FinalTrainRadiomics")

# =============================================================================
# Protocol Config (Optimized for Radiomics 2-Layer MLP)
# =============================================================================
PROTOCOL = {
    "lr": 5e-5,                 
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 220,
    "num_workers": 4,
    "use_cosine": True,
    "warmup_epochs": 10,        
    "use_cache": True,

    "global_threshold": 0.5,

    "roi_size": [32, 64, 64],
    "window_width": 400.0,
    "window_level": 40.0,

    "latency_warmup_steps": 10,
    "cohort_mode": "intersection",

    "rad_input_features": 112,
    "rad_hidden_features": 64,
    "num_classes": 2,
    "dropout_rate": 0.4,       

    "seed": 2026,
}

# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"[Seed Lock] {seed}")

def calculate_fixed_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    auc = 0.5
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))

    sens = spec = 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        pass

    return {
        "AUC": auc,
        "ACC": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Threshold": float(threshold),
    }

def compute_class_weights(loader, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        labels = batch["label"].detach().cpu().numpy().astype(int)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)

# =============================================================================
# Radiomics-only Model (Optimized 2-Layer with BatchNorm)
# =============================================================================
class RadiomicsOnlyModel(nn.Module):
    """
    Input dict contract:
        {"rad_features": Tensor}
    Training output:
        {"fusion": logits, "img": None, "rad": None}
    Eval output:
        (logits, None, None)
    """
    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        x = batch["rad_features"]
        logits = self.mlp(x)
        if self.training:
            return {"fusion": logits, "img": None, "rad": None}
        return (logits, None, None)

# =============================================================================
# Train / Eval / Latency
# =============================================================================
def train_one_epoch(model, loader, optimizer, crit_main, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch["label"].to(device).long()
        optimizer.zero_grad(set_to_none=True)

        out = model({
            "rad_features": batch["rad_features"].to(device),
        })

        logits = out["fusion"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
        loss = crit_main(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate_split(model, loader, device, threshold: float):
    model.eval()
    y_true, y_prob, y_ids = [], [], []

    for batch in loader:
        labels = batch["label"].to(device).long()

        out = model({
            "rad_features": batch["rad_features"].to(device),
        })
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=1)[:, 1]

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

        ids = batch.get("id", None)
        if ids is None:
            y_ids.extend([f"unknown_{i}" for i in range(len(labels))])
        elif isinstance(ids, list):
            y_ids.extend([str(x) for x in ids])
        else:
            y_ids.extend([str(ids)] * len(labels))

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    metrics = calculate_fixed_metrics(y_true_arr, y_prob_arr, threshold)

    return metrics, (y_true_arr, y_prob_arr, np.asarray(y_ids, dtype=object))

@torch.no_grad()
def compute_inference_latency_bs1(model, loader, device, warmup_steps: int = 10) -> float:
    model.eval()
    if device.type == "cuda":
        steps = 0
        for batch in loader:
            if steps >= warmup_steps:
                break
            _ = model({"rad_features": batch["rad_features"].to(device)})
            torch.cuda.synchronize(device)
            steps += 1

    total_time = 0.0
    total_samples = 0
    for batch in loader:
        rads = batch["rad_features"].to(device)
        bs = int(rads.size(0))

        for i in range(bs):
            single = {"rad_features": rads[i:i + 1]}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(single)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += (time.perf_counter() - t0)
            total_samples += 1

    return float(total_time / total_samples) if total_samples > 0 else 0.0

# =============================================================================
# Args
# =============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final Global Training Script (Radiomics Only - 2 Layer MLP)")

    base = Path(os.environ.get("CPLAN_DATA_DIR", "/root/autodl-tmp/data"))
    rad = base / "radiomics_features"

    p.add_argument("--internal_csv", type=Path, default=base / "internal group.csv")
    p.add_argument("--external_csv", type=Path, default=base / "external label.csv")
    p.add_argument("--trainval_img_dir", type=Path, default=base / "trainandval crop 3d")
    p.add_argument("--internal_test_img_dir", type=Path, default=base / "internal test crop 3d")
    p.add_argument("--external_test_img_dir", type=Path, default=base / "external LN 294 crop 3d")
    p.add_argument("--rad_trainval_csv", type=Path, default=rad / "radiomics_internal_trainval_sel.csv")
    p.add_argument("--rad_internal_test_csv", type=Path, default=rad / "radiomics_internal_test_sel.csv")
    p.add_argument("--rad_external_test_csv", type=Path, default=rad / "radiomics_external_test_sel.csv")

    p.add_argument("--output_dir", type=Path, default=Path("./checkpoints_final_radiomics_model"))
    return p.parse_args()

def apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    for k, v in PROTOCOL.items():
        setattr(args, k, v)
    return args

# =============================================================================
# Main
# =============================================================================
def main():
    args = apply_protocol(get_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 80)
    log.info("STARTING FINAL GLOBAL TRAINING (RADIOMICS ONLY - OPTIMIZED)")
    log.info(f"Target Epochs: {args.epochs}")
    log.info(f"Prospective Evaluation Threshold: {args.global_threshold}")
    log.info("=" * 80)

    dataloaders = get_final_radiomics_dataloaders(args)
    cls_w = compute_class_weights(dataloaders["train"], num_classes=args.num_classes).to(device)
    log.info(f"Global Class Weights: {cls_w.detach().cpu().numpy().round(4).tolist()}")

    model = RadiomicsOnlyModel(
        in_features=args.rad_input_features,
        hidden=args.rad_hidden_features,
        num_classes=args.num_classes,
        dropout=float(args.dropout_rate),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit_main = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05) 

    scheduler = None
    if args.use_cosine:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    log.info("--- Training Phase ---")
    for epoch in range(1, args.epochs + 1):
        if args.use_cosine and epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_loss = train_one_epoch(model, dataloaders["train"], optimizer, crit_main, device)

        if args.use_cosine and epoch > args.warmup_epochs and scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            log.info(f"Epoch {epoch:03d}/{args.epochs} | LR: {cur_lr:.6f} | Global Train Loss: {train_loss:.4f}")

    save_path = args.output_dir / "final_radiomics_only.pth"
    torch.save(model.state_dict(), save_path)
    log.info(f"Training Complete. Weights locked and saved to: {save_path}")

    log.info("=" * 80)
    log.info(f"PROSPECTIVE EVALUATION (Threshold = {args.global_threshold})")
    log.info("=" * 80)

    metrics_repo = []
    pred_repo = []

    splits = [
        ("Global Train", "train"),
        ("Internal Test", "internal_test"),
        ("External Test", "external_test")
    ]

    lat_val = None
    if "external_test" in dataloaders and dataloaders["external_test"] is not None:
        log.info("[Timing] Benchmarking inference latency (BS=1, warmup)...")
        lat_val = compute_inference_latency_bs1(
            model, dataloaders["external_test"], device, warmup_steps=int(args.latency_warmup_steps)
        )
        log.info(f"[Timing] Inference Latency: {lat_val:.4f} s/case")

    for name, key in splits:
        if key in dataloaders and dataloaders[key] is not None:
            m, (y, p, ids) = evaluate_split(model, dataloaders[key], device, args.global_threshold)
            m["Dataset"] = name
            if name == "External Test" and lat_val is not None:
                m["Inference Latency (s)"] = lat_val
            metrics_repo.append(m)

            for _id, lbl, pr in zip(ids, y, p):
                pred_repo.append({"Dataset": name, "ID": str(_id), "Label": int(lbl), "Probability": float(pr)})

            log.info(
                f"[{name:<14}] AUC: {m['AUC']:.4f} | ACC: {m['ACC']:.4f} | Sens: {m['Sensitivity']:.4f} | Spec: {m['Specificity']:.4f}"
            )

    if metrics_repo:
        df_metrics = pd.DataFrame(metrics_repo)
        cols = ["Dataset", "AUC", "ACC", "Sensitivity", "Specificity", "Threshold"]
        if "Inference Latency (s)" in df_metrics.columns:
            cols.append("Inference Latency (s)")
        df_metrics = df_metrics[cols]
        df_metrics.to_csv(args.output_dir / "final_evaluation_metrics.csv", index=False)
        log.info(f"Metrics saved to {args.output_dir / 'final_evaluation_metrics.csv'}")

    if pred_repo:
        pd.DataFrame(pred_repo).to_csv(args.output_dir / "final_predictions.csv", index=False)

    log.info("Final Pipeline Execution Terminated Successfully.")

if __name__ == "__main__":
    main()