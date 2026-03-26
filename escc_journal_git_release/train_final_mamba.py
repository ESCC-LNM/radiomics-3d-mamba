"""
Final Global Training Script for Mamba Fusion (Radiomics_3D_Mamba)

Methodological Integrity:
1. Fixed-Epoch Training: hard stop at Epoch 220. No validation set, no early stopping.
2. Prospective Evaluation: fixed global threshold (0.5).
3. Test Set Isolation: evaluated only once post-training.
4. Image input uses 2 channels: [CT, binary mask].
5. Fusion strategy uses learnable convex gating from mamba_fusion_model.py.
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
    from data_pipeline_final import get_final_fusion_dataloaders
    from mamba_fusion_model import Mamba_Fusion_Model
except ImportError as e:
    sys.stderr.write(f"Import Error: {e}\nEnsure data_pipeline_final.py and mamba_fusion_model.py are in scope.\n")
    sys.exit(1)


LOG_FILENAME = "train_final_mamba.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILENAME, mode="w")],
)
log = logging.getLogger("FinalTrain")


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

    "mamba_embed_dim": 128,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "image_in_channels": 2,
    "rad_input_features": 112,
    "rad_hidden_features": 64,
    "fusion_hidden_features": 64,
    "num_classes": 2,
    "dropout_rate": 0.4,

    # Final fixed training setting (no ablation presets).
    "rad_scale": 0.70,
    "aux_img_weight": 0.25,
    "aux_rad_weight": 0.00,
    "disable_aux": False,

    "seed": 2026,
}


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
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        pass

    return {"AUC": auc, "ACC": acc, "Sensitivity": sens, "Specificity": spec, "Threshold": float(threshold)}


def compute_class_weights(loader, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        labels = batch["label"].detach().cpu().numpy().astype(int)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def build_image_tensor(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if "image" not in batch:
        raise KeyError("Batch is missing 'image'.")
    if "mask" not in batch:
        raise KeyError("Batch is missing 'mask'. Ensure mask paths and data_pipeline_final.py are updated.")

    img = batch["image"].to(device)
    msk = batch["mask"].to(device)
    msk = (msk > 0).to(dtype=img.dtype)

    if img.dim() != 5 or msk.dim() != 5:
        raise ValueError(f"Expected 5D tensors (B,C,D,H,W), got image={tuple(img.shape)}, mask={tuple(msk.shape)}")
    if img.shape[0] != msk.shape[0] or img.shape[2:] != msk.shape[2:]:
        raise ValueError(f"Image/mask shape mismatch: image={tuple(img.shape)}, mask={tuple(msk.shape)}")

    return torch.cat([img, msk], dim=1)


def train_one_epoch(model, loader, optimizer, crit_main, crit_aux, device, use_aux: bool, w_img: float,
                    w_rad: float) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch["label"].to(device).long()
        optimizer.zero_grad(set_to_none=True)

        img_x = build_image_tensor(batch, device)
        out = model({
            "image": img_x,
            "rad_features": batch["rad_features"].to(device),
        })

        if use_aux and isinstance(out, dict):
            loss = crit_main(out["fusion"], labels)
            if out.get("img") is not None:
                loss = loss + float(w_img) * crit_aux(out["img"], labels)
            if out.get("rad") is not None:
                loss = loss + float(w_rad) * crit_aux(out["rad"], labels)
        else:
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
        img_x = build_image_tensor(batch, device)
        out = model({
            "image": img_x,
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
            img_x = build_image_tensor(batch, device)
            _ = model({"image": img_x, "rad_features": batch["rad_features"].to(device)})
            torch.cuda.synchronize(device)
            steps += 1

    total_time = 0.0
    total_samples = 0
    for batch in loader:
        imgs = build_image_tensor(batch, device)
        rads = batch["rad_features"].to(device)
        bs = int(imgs.size(0))

        for i in range(bs):
            single = {"image": imgs[i:i + 1], "rad_features": rads[i:i + 1]}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(single)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += (time.perf_counter() - t0)
            total_samples += 1

    return float(total_time / total_samples) if total_samples > 0 else 0.0


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final Global Training Script")

    base = Path(os.environ.get("CPLAN_DATA_DIR", "/root/autodl-tmp/data"))
    rad = base / "radiomics_features"

    p.add_argument("--internal_csv", type=Path, default=base / "internal group.csv")
    p.add_argument("--external_csv", type=Path, default=base / "external label.csv")
    p.add_argument("--trainval_img_dir", type=Path, default=base / "trainandval crop 3d")
    p.add_argument("--internal_test_img_dir", type=Path, default=base / "internal test crop 3d")
    p.add_argument("--external_test_img_dir", type=Path, default=base / "external LN 294 crop 3d")
    p.add_argument("--trainval_mask_dir", type=Path, default=base / "trainandval masks")
    p.add_argument("--internal_test_mask_dir", type=Path, default=base / "internal test masks")
    p.add_argument("--external_test_mask_dir", type=Path, default=base / "external test masks")
    p.add_argument("--rad_trainval_csv", type=Path, default=rad / "radiomics_internal_trainval_sel.csv")
    p.add_argument("--rad_internal_test_csv", type=Path, default=rad / "radiomics_internal_test_sel.csv")
    p.add_argument("--rad_external_test_csv", type=Path, default=rad / "radiomics_external_test_sel.csv")

    p.add_argument("--output_dir", type=Path, default=Path("./checkpoints_final_model"))
    return p.parse_args()


def apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    for k, v in PROTOCOL.items():
        setattr(args, k, v)
    return args


def main():
    args = apply_protocol(get_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("CUDA not detected. Native Mamba kernels will fail.")

    log.info("=" * 80)
    log.info("STARTING FINAL GLOBAL TRAINING (NO K-FOLD, NO VALIDATION)")
    log.info(f"Target Epochs: {args.epochs}")
    log.info(f"Prospective Evaluation Threshold: {args.global_threshold}")
    log.info(
        f"Final Fusion Config | Aux={(not args.disable_aux)} "
        f"RadScale={args.rad_scale} AuxImgW={args.aux_img_weight} "
        f"AuxRadW={args.aux_rad_weight} ImgInCh={args.image_in_channels}"
    )
    log.info("=" * 80)

    dataloaders = get_final_fusion_dataloaders(args)
    cls_w = compute_class_weights(dataloaders["train"], num_classes=args.num_classes).to(device)
    log.info(f"Global Class Weights: {cls_w.detach().cpu().numpy().round(4).tolist()}")

    model = Mamba_Fusion_Model(
        mamba_embed_dim=args.mamba_embed_dim,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        image_in_channels=args.image_in_channels,
        rad_input_features=args.rad_input_features,
        rad_hidden_features=args.rad_hidden_features,
        fusion_hidden_features=args.fusion_hidden_features,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        use_aux_cls=(not args.disable_aux),
        rad_scale=args.rad_scale,
        require_cuda=(device.type == "cuda"),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit_main = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    crit_aux = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)

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

        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, crit_main, crit_aux, device,
            not args.disable_aux, args.aux_img_weight, args.aux_rad_weight
        )

        if args.use_cosine and epoch > args.warmup_epochs and scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            log.info(f"Epoch {epoch:03d}/{args.epochs} | LR: {cur_lr:.6f} | Global Train Loss: {train_loss:.4f}")

    save_path = args.output_dir / "final_mamba_fusion.pth"
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
        ("External Test", "external_test"),
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
                f"[{name:<14}] AUC: {m['AUC']:.4f} | ACC: {m['ACC']:.4f} | "
                f"Sens: {m['Sensitivity']:.4f} | Spec: {m['Specificity']:.4f}"
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
