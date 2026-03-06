from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.optim import Adam


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
def _setup_imports():
    try:
        if __package__:
            from .data_pipeline_final import get_final_fusion_dataloaders, peek_radiomics_dim  # type: ignore
            from .mamba_fusion_model import Mamba_Fusion_Model  # type: ignore
            return get_final_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model
    except Exception:
        pass

    current_file = Path(__file__).resolve()
    mod_dir = current_file.parent
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    from data_pipeline_final import get_final_fusion_dataloaders, peek_radiomics_dim  # noqa
    from mamba_fusion_model import Mamba_Fusion_Model  # noqa
    return get_final_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model


get_final_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model = _setup_imports()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mamba_fusion_final_v2")


DEFAULTS: Dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 3e-5,
    "batch_size": 16,
    "epochs": 220,
    "num_workers": 4,
    "use_cosine": True,
    "warmup_epochs": 10,
    "roi_size": [32, 64, 64],
    "window_width": 400.0,
    "window_level": 40.0,
    "mamba_embed_dim": 128,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "rad_hidden_features": 64,
    "fusion_hidden_features": 64,
    "num_classes": 2,
    "dropout_rate": 0.3,
    "rad_scale": 0.3,
    "aux_img_weight": 0.1,
    "aux_rad_weight": 0.05,
    "disable_aux": False,
    "disable_gate": False,
    "seed": 42,
}


# -----------------------------------------------------------------------------
# Repro / args
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final global training with parameters imported from CV tuning summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tuning_summary_json", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("./outputs/final_model"))

    p.add_argument("--internal_csv", type=Path, default=Path("./data3D/internal_group.csv"))
    p.add_argument("--external_csv", type=Path, default=Path("./data3D/external_label.csv"))
    p.add_argument("--trainval_img_dir", type=Path, default=Path("./data3D/trainval_images"))
    p.add_argument("--internal_test_img_dir", type=Path, default=Path("./data3D/internal_test_images"))
    p.add_argument("--external_test_img_dir", type=Path, default=Path("./data3D/external_test_images"))
    p.add_argument("--rad_trainval_csv", type=Path, default=Path("./outputs/selected_features/final/radiomics_internal_trainval_sel.csv"))
    p.add_argument("--rad_internal_test_csv", type=Path, default=Path("./outputs/selected_features/final/radiomics_internal_test_sel.csv"))
    p.add_argument("--rad_external_test_csv", type=Path, default=Path("./outputs/selected_features/final/radiomics_external_test_sel.csv"))

    p.add_argument("--id_col", type=str, default="ID")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--group_col", type=str, default="group")
    p.add_argument("--patient_id_col", type=str, default="patient_id")
    p.add_argument("--train_group_value", type=str, default="train")
    p.add_argument("--internal_test_group_value", type=str, default="test")
    p.add_argument("--nifti_exts", type=str, nargs="+", default=[".nii.gz", ".nii", ".nii(1).gz"])

    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--use_cosine", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=DEFAULTS["warmup_epochs"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--no_cache", dest="use_cache", action="store_false")
    p.set_defaults(use_cache=True, use_cosine=DEFAULTS["use_cosine"])

    p.add_argument("--roi_size", type=int, nargs=3, default=DEFAULTS["roi_size"])
    p.add_argument("--window_width", type=float, default=DEFAULTS["window_width"])
    p.add_argument("--window_level", type=float, default=DEFAULTS["window_level"])

    p.add_argument("--mamba_embed_dim", type=int, default=DEFAULTS["mamba_embed_dim"])
    p.add_argument("--mamba_d_state", type=int, default=DEFAULTS["mamba_d_state"])
    p.add_argument("--mamba_d_conv", type=int, default=DEFAULTS["mamba_d_conv"])
    p.add_argument("--mamba_expand", type=int, default=DEFAULTS["mamba_expand"])
    p.add_argument("--rad_hidden_features", type=int, default=DEFAULTS["rad_hidden_features"])
    p.add_argument("--fusion_hidden_features", type=int, default=DEFAULTS["fusion_hidden_features"])
    p.add_argument("--num_classes", type=int, default=DEFAULTS["num_classes"])
    p.add_argument("--dropout_rate", type=float, default=DEFAULTS["dropout_rate"])
    p.add_argument("--rad_scale", type=float, default=DEFAULTS["rad_scale"])
    p.add_argument("--aux_img_weight", type=float, default=DEFAULTS["aux_img_weight"])
    p.add_argument("--aux_rad_weight", type=float, default=DEFAULTS["aux_rad_weight"])
    p.add_argument("--disable_aux", action="store_true", default=DEFAULTS["disable_aux"])
    p.add_argument("--disable_gate", action="store_true", default=DEFAULTS["disable_gate"])

    p.add_argument("--return_ids", action="store_true")
    p.add_argument("--anonymize_ids", action="store_true")
    return p.parse_args()


def load_tuning_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_tuning_summary(args: argparse.Namespace, summary: Dict[str, Any]) -> argparse.Namespace:
    cfg = summary.get("config", {})
    for k in [
        "lr", "weight_decay", "batch_size", "use_cosine", "warmup_epochs",
        "dropout_rate", "rad_scale", "mamba_embed_dim", "mamba_d_state",
        "mamba_d_conv", "mamba_expand", "rad_hidden_features",
        "fusion_hidden_features", "disable_aux", "disable_gate", "seed",
    ]:
        if k in cfg:
            setattr(args, k, cfg[k])
    args.epochs = int(summary.get("recommended_epochs", args.epochs))
    args.global_threshold = float(summary.get("recommended_threshold", 0.5))
    args.rad_input_features = int(summary.get("recommended_rad_input_features", peek_radiomics_dim(Path(args.rad_trainval_csv))))
    return args


# -----------------------------------------------------------------------------
# Metrics / eval
# -----------------------------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    auc = 0.5
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    y_pred = (y_prob >= float(threshold)).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    sens = 0.0
    spec = 0.0
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


def train_one_epoch(model, loader, optimizer, criterion_main, criterion_aux, device, use_aux: bool, aux_img_weight: float, aux_rad_weight: float) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch["label"].to(device).long()
        optimizer.zero_grad(set_to_none=True)
        outputs = model({"image": batch["image"].to(device), "rad_features": batch["rad_features"].to(device)})
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if use_aux and isinstance(outputs, dict):
            loss = criterion_main(outputs["fusion"], labels)
            if outputs.get("img") is not None:
                loss = loss + float(aux_img_weight) * criterion_aux(outputs["img"], labels)
            if outputs.get("rad") is not None:
                loss = loss + float(aux_rad_weight) * criterion_aux(outputs["rad"], labels)
        else:
            logits = outputs["fusion"] if isinstance(outputs, dict) else outputs
            loss = criterion_main(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_split(model, loader, device, threshold: float):
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    y_ids: List[str] = []
    for batch in loader:
        logits = model({"image": batch["image"].to(device), "rad_features": batch["rad_features"].to(device)})
        if isinstance(logits, tuple):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits["fusion"]
        probs = torch.softmax(logits, dim=1)[:, 1]
        labels = batch["label"].detach().cpu().numpy().astype(int)
        y_true.extend(labels.tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        ids = batch.get("id", None)
        if ids is None:
            y_ids.extend([f"idx_{len(y_ids) + i}" for i in range(len(labels))])
        elif isinstance(ids, list):
            y_ids.extend([str(x) for x in ids])
        else:
            y_ids.extend([str(ids)] * len(labels))
    arr_true = np.asarray(y_true, dtype=int)
    arr_prob = np.asarray(y_prob, dtype=float)
    metrics = calculate_metrics(arr_true, arr_prob, threshold=threshold)
    return metrics, arr_true, arr_prob, np.asarray(y_ids, dtype=object)


@torch.no_grad()
def compute_inference_latency_bs1(model, loader, device, warmup_steps: int = 10) -> float:
    model.eval()
    if device.type == "cuda":
        steps = 0
        for batch in loader:
            if steps >= warmup_steps:
                break
            _ = model({"image": batch["image"].to(device), "rad_features": batch["rad_features"].to(device)})
            torch.cuda.synchronize(device)
            steps += 1
    total_time = 0.0
    total_samples = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        rads = batch["rad_features"].to(device)
        bs = int(imgs.size(0))
        for i in range(bs):
            single = {"image": imgs[i:i+1], "rad_features": rads[i:i+1]}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(single)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time += time.perf_counter() - t0
            total_samples += 1
    return float(total_time / total_samples) if total_samples > 0 else 0.0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    summary = load_tuning_summary(Path(args.tuning_summary_json))
    args = apply_tuning_summary(args, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Final training starts | device=%s | epochs=%d | threshold=%.6f | rad_dim=%d", device, int(args.epochs), float(args.global_threshold), int(args.rad_input_features))
    log.info("Final stage uses global train pool only, and evaluates internal/external test once after training.")

    dataloaders = get_final_fusion_dataloaders(args)
    class_weights = compute_class_weights(dataloaders["train"], num_classes=int(args.num_classes)).to(device)

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

    optimizer = Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion_main = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_aux = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    scheduler = None
    if bool(args.use_cosine):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) - int(args.warmup_epochs)))

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion_main,
            criterion_aux,
            device,
            not bool(args.disable_aux),
            float(args.aux_img_weight),
            float(args.aux_rad_weight),
        )
        if bool(args.use_cosine):
            if epoch <= int(args.warmup_epochs):
                warmup_lr = float(args.lr) * epoch / max(1, int(args.warmup_epochs))
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
            elif scheduler is not None:
                scheduler.step()
        if epoch % 10 == 0 or epoch == int(args.epochs):
            log.info("Epoch %03d/%03d | lr=%.7f | train_loss=%.4f", epoch, int(args.epochs), optimizer.param_groups[0]["lr"], train_loss)

    save_path = args.output_dir / "final_mamba_fusion.pth"
    torch.save(model.state_dict(), save_path)
    log.info("Saved final model: %s", save_path)

    metrics_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    latency = compute_inference_latency_bs1(model, dataloaders["external_test"], device=device, warmup_steps=10)
    for split_name, key in [("Global Train", "train"), ("Internal Test", "internal_test"), ("External Test", "external_test")]:
        metrics, y_true, y_prob, y_ids = evaluate_split(model, dataloaders[key], device=device, threshold=float(args.global_threshold))
        row = {"Dataset": split_name, **metrics}
        if split_name == "External Test":
            row["Inference Latency (s)"] = float(latency)
        metrics_rows.append(row)
        for _id, yt, yp in zip(y_ids.tolist(), y_true.tolist(), y_prob.tolist()):
            pred_rows.append({"Dataset": split_name, "ID": str(_id), "Label": int(yt), "Probability": float(yp)})
        log.info("[%s] AUC=%.4f | ACC=%.4f | Sens=%.4f | Spec=%.4f | Thr=%.4f", split_name, metrics["AUC"], metrics["ACC"], metrics["Sensitivity"], metrics["Specificity"], metrics["Threshold"])

    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "final_evaluation_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "final_predictions.csv", index=False)

    used_protocol = {
        "epochs": int(args.epochs),
        "threshold": float(args.global_threshold),
        "rad_input_features": int(args.rad_input_features),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "dropout_rate": float(args.dropout_rate),
        "rad_scale": float(args.rad_scale),
        "tuning_summary_json": str(Path(args.tuning_summary_json)),
    }
    with open(args.output_dir / "final_run_protocol.json", "w", encoding="utf-8") as f:
        json.dump(used_protocol, f, indent=2, ensure_ascii=False)

    log.info("Saved final metrics, predictions, and run protocol to %s", args.output_dir)


if __name__ == "__main__":
    main(get_args())
