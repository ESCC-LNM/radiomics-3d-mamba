from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.optim import Adam


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
def _setup_imports():
    try:
        if __package__:
            from .data_pipeline_fusion import get_fusion_dataloaders, peek_radiomics_dim  # type: ignore
            from .mamba_fusion_model import Mamba_Fusion_Model  # type: ignore
            return get_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model
    except Exception:
        pass

    current_file = Path(__file__).resolve()
    mod_dir = current_file.parent
    if str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))
    from data_pipeline_fusion import get_fusion_dataloaders, peek_radiomics_dim  # noqa
    from mamba_fusion_model import Mamba_Fusion_Model  # noqa
    return get_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model


get_fusion_dataloaders, peek_radiomics_dim, Mamba_Fusion_Model = _setup_imports()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mamba_fusion_tuning_v2")


DEFAULT_CFG: Dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 3e-5,
    "batch_size": 16,
    "epochs": 220,
    "n_splits": 5,
    "num_workers": 4,
    "min_spec": 0.70,
    "use_cosine": True,
    "warmup_epochs": 10,
    "early_stop": True,
    "patience": 30,
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
    "use_cache": True,
    "seed": 42,
    "nifti_exts": [".nii.gz", ".nii", ".nii(1).gz"],
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
    try:
        from monai.utils import set_determinism  # type: ignore
        set_determinism(seed=seed)
    except Exception:
        pass


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5-fold patient-safe tuning for Mamba fusion. Outputs params only; no final test metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", type=Path, default=Path("./outputs/tuning_runs"))
    p.add_argument("--internal_csv", type=Path, default=Path("./data3D/internal_group.csv"))
    p.add_argument("--external_csv", type=Path, default=Path("./data3D/external_label.csv"))
    p.add_argument("--trainval_img_dir", type=Path, default=Path("./data3D/trainval_images"))
    p.add_argument("--internal_test_img_dir", type=Path, default=Path("./data3D/internal_test_images"))
    p.add_argument("--external_test_img_dir", type=Path, default=Path("./data3D/external_test_images"))
    p.add_argument("--split_manifest_csv", type=Path, required=True)
    p.add_argument("--rad_root_dir", type=Path, required=True, help="Folder containing fold01..fold05 selected radiomics CSVs.")

    p.add_argument("--id_col", type=str, default="ID")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--group_col", type=str, default="group")
    p.add_argument("--patient_id_col", type=str, default="patient_id")
    p.add_argument("--train_group_value", type=str, default="train")
    p.add_argument("--internal_test_group_value", type=str, default="test")

    p.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["weight_decay"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CFG["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    p.add_argument("--n_splits", type=int, default=DEFAULT_CFG["n_splits"])
    p.add_argument("--num_workers", type=int, default=DEFAULT_CFG["num_workers"])
    p.add_argument("--min_spec", type=float, default=DEFAULT_CFG["min_spec"])
    p.add_argument("--use_cosine", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=DEFAULT_CFG["warmup_epochs"])
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--patience", type=int, default=DEFAULT_CFG["patience"])
    p.add_argument("--run_fold", type=int, default=None, help="1-based fold id, optional.")
    p.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--no_cache", dest="use_cache", action="store_false")
    p.set_defaults(use_cache=DEFAULT_CFG["use_cache"], use_cosine=DEFAULT_CFG["use_cosine"], early_stop=DEFAULT_CFG["early_stop"])

    p.add_argument("--roi_size", type=int, nargs=3, default=DEFAULT_CFG["roi_size"])
    p.add_argument("--window_width", type=float, default=DEFAULT_CFG["window_width"])
    p.add_argument("--window_level", type=float, default=DEFAULT_CFG["window_level"])
    p.add_argument("--nifti_exts", type=str, nargs="+", default=DEFAULT_CFG["nifti_exts"])

    p.add_argument("--mamba_embed_dim", type=int, default=DEFAULT_CFG["mamba_embed_dim"])
    p.add_argument("--mamba_d_state", type=int, default=DEFAULT_CFG["mamba_d_state"])
    p.add_argument("--mamba_d_conv", type=int, default=DEFAULT_CFG["mamba_d_conv"])
    p.add_argument("--mamba_expand", type=int, default=DEFAULT_CFG["mamba_expand"])
    p.add_argument("--rad_hidden_features", type=int, default=DEFAULT_CFG["rad_hidden_features"])
    p.add_argument("--fusion_hidden_features", type=int, default=DEFAULT_CFG["fusion_hidden_features"])
    p.add_argument("--num_classes", type=int, default=DEFAULT_CFG["num_classes"])
    p.add_argument("--dropout_rate", type=float, default=DEFAULT_CFG["dropout_rate"])
    p.add_argument("--rad_scale", type=float, default=DEFAULT_CFG["rad_scale"])
    p.add_argument("--aux_img_weight", type=float, default=DEFAULT_CFG["aux_img_weight"])
    p.add_argument("--aux_rad_weight", type=float, default=DEFAULT_CFG["aux_rad_weight"])
    p.add_argument("--disable_aux", action="store_true", default=DEFAULT_CFG["disable_aux"])
    p.add_argument("--disable_gate", action="store_true", default=DEFAULT_CFG["disable_gate"])
    p.add_argument("--return_ids", action="store_true")
    p.add_argument("--anonymize_ids", action="store_true")
    return p.parse_args()


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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_spec: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
    except Exception:
        return 0.5
    spec = 1.0 - fpr
    sens = tpr
    ok = spec >= float(min_spec)
    if np.any(ok):
        diff = np.abs(sens[ok] - spec[ok])
        idx_rel = int(np.argmin(diff))
        return float(thr[ok][idx_rel])
    youden = sens - fpr
    idx = int(np.argmax(youden))
    return float(thr[idx])


def compute_class_weights_from_loader(loader, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        labels = batch["label"].detach().cpu().numpy().astype(int)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def validate(model: nn.Module, loader, criterion_val, device: torch.device, threshold: float = 0.5, return_raw: bool = False):
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_ids: List[str] = []

    for batch in loader:
        labels = batch["label"].to(device).long()
        logits = model({"image": batch["image"].to(device), "rad_features": batch["rad_features"].to(device)})
        if isinstance(logits, tuple):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits["fusion"]
        loss = criterion_val(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += float(loss.item())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        ids = batch.get("id", None)
        if ids is None:
            all_ids.extend([f"idx_{len(all_ids) + i}" for i in range(len(labels))])
        elif isinstance(ids, list):
            all_ids.extend([str(x) for x in ids])
        else:
            all_ids.extend([str(ids)] * len(labels))

    avg_loss = total_loss / max(len(loader), 1)
    y_true = np.asarray(all_labels, dtype=int)
    y_prob = np.asarray(all_probs, dtype=float)
    if return_raw:
        return avg_loss, y_true, y_prob, np.asarray(all_ids, dtype=object)
    return avg_loss, calculate_metrics(y_true, y_prob, threshold=float(threshold))


def train_one_epoch(model: nn.Module, loader, optimizer: Adam, criterion_main, criterion_aux, device: torch.device, use_aux: bool, aux_img_weight: float, aux_rad_weight: float) -> float:
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


# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
def apply_fold_radiomics_paths(args: argparse.Namespace, fold_id_1based: int) -> None:
    fold_dir = Path(args.rad_root_dir) / f"fold{fold_id_1based:02d}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Expected fold directory not found: {fold_dir}")
    args.rad_trainval_csv = fold_dir / "radiomics_internal_trainval_sel.csv"
    args.rad_internal_test_csv = fold_dir / "radiomics_internal_test_sel.csv"
    args.rad_external_test_csv = fold_dir / "radiomics_external_test_sel.csv"
    args.rad_input_features = int(peek_radiomics_dim(args.rad_trainval_csv))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = [int(args.run_fold) - 1] if args.run_fold is not None else list(range(int(args.n_splits)))
    log.info("Starting CV tuning | device=%s | folds=%s", device, [f + 1 for f in folds])
    log.info("Test sets are NOT used in tuning. Output is only parameters + OOF validation artifacts.")

    all_fold_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []

    for fold_idx in folds:
        fold_id = fold_idx + 1
        apply_fold_radiomics_paths(args, fold_id)
        args.fold_idx = fold_idx
        log.info("================ Fold %d/%d | rad_dim=%d ================", fold_id, int(args.n_splits), int(args.rad_input_features))

        dataloaders = get_fusion_dataloaders(args)
        class_weights = compute_class_weights_from_loader(dataloaders["train"], num_classes=int(args.num_classes)).to(device)

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
        criterion_val = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)

        scheduler = None
        if bool(args.use_cosine):
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) - int(args.warmup_epochs)))

        best_val_auc = -np.inf
        best_epoch = 0
        best_ckpt = args.output_dir / f"fold{fold_id:02d}_best.pth"
        no_improve = 0

        for epoch in range(1, int(args.epochs) + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=dataloaders["train"],
                optimizer=optimizer,
                criterion_main=criterion_main,
                criterion_aux=criterion_aux,
                device=device,
                use_aux=not bool(args.disable_aux),
                aux_img_weight=float(args.aux_img_weight),
                aux_rad_weight=float(args.aux_rad_weight),
            )

            if bool(args.use_cosine):
                if epoch <= int(args.warmup_epochs):
                    warmup_lr = float(args.lr) * epoch / max(1, int(args.warmup_epochs))
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                elif scheduler is not None:
                    scheduler.step()

            val_loss, val_metrics = validate(model, dataloaders["val"], criterion_val, device=device, threshold=0.5)
            current_auc = float(val_metrics["AUC"])
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "Fold %02d | Epoch %03d/%03d | lr=%.7f | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f",
                fold_id, epoch, int(args.epochs), current_lr, train_loss, val_loss, current_auc,
            )

            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_epoch = int(epoch)
                no_improve = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                no_improve += 1

            if bool(args.early_stop) and no_improve >= int(args.patience):
                log.info("Fold %02d early-stopped at epoch %03d (best_epoch=%03d)", fold_id, epoch, best_epoch)
                break

        if not best_ckpt.exists():
            raise RuntimeError(f"Best checkpoint missing for fold {fold_id}: {best_ckpt}")

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
        best_model.load_state_dict(torch.load(best_ckpt, map_location=device))

        _, y_true, y_prob, y_ids = validate(best_model, dataloaders["val"], criterion_val, device=device, return_raw=True)
        fold_thr = find_optimal_threshold(y_true, y_prob, min_spec=float(args.min_spec))
        fold_metrics = calculate_metrics(y_true, y_prob, threshold=fold_thr)
        fold_metrics.update(
            {
                "fold": fold_id,
                "best_epoch": int(best_epoch),
                "rad_input_features": int(args.rad_input_features),
                "train_samples": int(len(dataloaders["train"].dataset)),
                "val_samples": int(len(dataloaders["val"].dataset)),
                "checkpoint": str(best_ckpt),
            }
        )
        all_fold_rows.append(fold_metrics)

        for _id, yt, yp in zip(y_ids.tolist(), y_true.tolist(), y_prob.tolist()):
            oof_rows.append({"fold": fold_id, "ID": str(_id), "Label": int(yt), "Probability": float(yp)})

        log.info(
            "Fold %02d complete | best_epoch=%03d | val_auc=%.4f | fold_threshold=%.4f",
            fold_id,
            best_epoch,
            fold_metrics["AUC"],
            fold_thr,
        )

    if not all_fold_rows:
        raise RuntimeError("No fold results were produced.")

    df_folds = pd.DataFrame(all_fold_rows).sort_values("fold").reset_index(drop=True)
    df_folds.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)

    df_oof = pd.DataFrame(oof_rows)
    df_oof.to_csv(args.output_dir / "cv_oof_predictions.csv", index=False)

    pooled_thr = find_optimal_threshold(df_oof["Label"].values, df_oof["Probability"].values, min_spec=float(args.min_spec))
    pooled_metrics = calculate_metrics(df_oof["Label"].values, df_oof["Probability"].values, threshold=pooled_thr)

    recommended_epochs = int(np.round(df_folds["best_epoch"].astype(float).median()))
    recommended_rad_dim = int(pd.Series(df_folds["rad_input_features"].astype(int)).mode().iloc[0])

    summary = {
        "n_folds_ran": int(len(df_folds)),
        "split_manifest_csv": str(Path(args.split_manifest_csv)),
        "rad_root_dir": str(Path(args.rad_root_dir)),
        "recommended_epochs": int(recommended_epochs),
        "recommended_threshold": float(pooled_thr),
        "recommended_rad_input_features": int(recommended_rad_dim),
        "threshold_strategy": "pooled_oof_balance_with_floor",
        "min_spec": float(args.min_spec),
        "pooled_oof_metrics": pooled_metrics,
        "fold_best_epoch_mean": float(df_folds["best_epoch"].astype(float).mean()),
        "fold_best_epoch_median": float(df_folds["best_epoch"].astype(float).median()),
        "config": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "use_cosine": bool(args.use_cosine),
            "warmup_epochs": int(args.warmup_epochs),
            "dropout_rate": float(args.dropout_rate),
            "rad_scale": float(args.rad_scale),
            "mamba_embed_dim": int(args.mamba_embed_dim),
            "mamba_d_state": int(args.mamba_d_state),
            "mamba_d_conv": int(args.mamba_d_conv),
            "mamba_expand": int(args.mamba_expand),
            "rad_hidden_features": int(args.rad_hidden_features),
            "fusion_hidden_features": int(args.fusion_hidden_features),
            "disable_aux": bool(args.disable_aux),
            "disable_gate": bool(args.disable_gate),
            "seed": int(args.seed),
        },
    }

    with open(args.output_dir / "cv_tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Saved fold metrics: %s", args.output_dir / "cv_fold_metrics.csv")
    log.info("Saved OOF predictions: %s", args.output_dir / "cv_oof_predictions.csv")
    log.info("Saved tuning summary: %s", args.output_dir / "cv_tuning_summary.json")
    log.info("Recommended epochs=%d | recommended threshold=%.6f | pooled OOF AUC=%.4f", recommended_epochs, pooled_thr, pooled_metrics["AUC"])


if __name__ == "__main__":
    main(get_args())
