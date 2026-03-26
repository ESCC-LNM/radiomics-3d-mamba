"""
Optional preliminary 5-fold development-stage tuning for the radiomics-3D Mamba fusion model.

This script is intentionally separated from the final manuscript-reported held-out
analysis. It operates only within the development cohort and is intended for
preliminary hyperparameter / epoch selection before final fixed-protocol retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from data.data_loading_cv_multimodal import build_cv_dataloaders
from utils.analysis_utils import (
    collect_environment_metadata,
    compute_binary_metrics,
    configure_logging,
    ensure_dir,
    get_optional,
    get_required,
    load_json,
    median_epoch_recommendation,
    save_json,
    select_device,
    set_global_seed,
    summarise_numeric,
)
from models.model_multimodal_mamba_fusion import MambaFusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional outer cross-validation for preliminary hyperparameter / epoch selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional override for the output directory.")
    parser.add_argument("--device", type=str, default=None, help="Optional explicit device string, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def build_model(cfg: Mapping[str, Any], radiomics_input_dim: int, device: torch.device) -> nn.Module:
    model = MambaFusionModel(
        image_in_channels=int(get_required(cfg, "model.image_in_channels")),
        mamba_embed_dim=int(get_required(cfg, "model.mamba_embed_dim")),
        mamba_d_state=int(get_required(cfg, "model.mamba_d_state")),
        mamba_d_conv=int(get_required(cfg, "model.mamba_d_conv")),
        mamba_expand=int(get_required(cfg, "model.mamba_expand")),
        radiomics_input_dim=int(radiomics_input_dim),
        radiomics_hidden_dim=int(get_required(cfg, "model.radiomics_hidden_dim")),
        fusion_hidden_dim=int(get_required(cfg, "model.fusion_hidden_dim")),
        num_classes=int(get_required(cfg, "model.num_classes")),
        dropout_rate=float(get_required(cfg, "model.dropout_rate")),
        use_auxiliary_image_head=bool(get_optional(cfg, "model.use_auxiliary_image_head", default=True)),
        require_cuda=(device.type == "cuda"),
    )
    return model.to(device)


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module):
    opt_name = str(get_required(cfg, "training.optimizer.name")).lower()
    lr = float(get_required(cfg, "training.optimizer.learning_rate"))
    weight_decay = float(get_optional(cfg, "training.optimizer.weight_decay", default=0.0))
    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("Unsupported optimizer. Use 'adam' or 'adamw'.")


def build_scheduler(cfg: Mapping[str, Any], optimizer, max_epochs: int):
    scheduler_name = str(get_optional(cfg, "training.scheduler.name", default="none")).lower()
    if scheduler_name in {"none", ""}:
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_epochs)
    if scheduler_name == "step":
        step_size = int(get_required(cfg, "training.scheduler.step_size"))
        gamma = float(get_required(cfg, "training.scheduler.gamma"))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError("Unsupported scheduler. Use 'none', 'cosine', or 'step'.")


def compute_class_weights(loader, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        labels = batch["label"].detach().cpu().numpy().astype(int)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def compute_loss(cfg: Mapping[str, Any], outputs: Mapping[str, torch.Tensor], labels: torch.Tensor, ce_loss: nn.Module) -> torch.Tensor:
    loss = ce_loss(outputs["fusion_logits"], labels)
    aux_cfg = get_optional(cfg, "training.auxiliary_loss", default={}) or {}
    if bool(aux_cfg.get("enable", True)):
        image_weight = float(aux_cfg.get("image_weight", 0.0))
        radiomics_weight = float(aux_cfg.get("radiomics_weight", 0.0))
        if outputs.get("image_logits") is not None and image_weight > 0.0:
            loss = loss + image_weight * ce_loss(outputs["image_logits"], labels)
        if outputs.get("radiomics_logits") is not None and radiomics_weight > 0.0:
            loss = loss + radiomics_weight * ce_loss(outputs["radiomics_logits"], labels)
    return loss


def _selection_score(selection_metric: str, val_metrics: Mapping[str, float], val_loss: float) -> float:
    metric = str(selection_metric).lower()
    if metric == "auc":
        return float(val_metrics["auc"])
    if metric == "balanced_accuracy":
        return float(val_metrics["balanced_accuracy"])
    if metric == "accuracy":
        return float(val_metrics["accuracy"])
    if metric == "negative_loss":
        return -float(val_loss)
    raise ValueError("Unsupported model-selection metric. Use auc, balanced_accuracy, accuracy, or negative_loss.")


def train_one_epoch(model: nn.Module, loader, optimizer, ce_loss, device: torch.device, cfg: Mapping[str, Any]) -> float:
    model.train()
    clip_norm = get_optional(cfg, "training.max_grad_norm", default=None)
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        rad = batch["rad_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model({"image": images, "rad_features": rad})
        loss = compute_loss(cfg, outputs, labels, ce_loss)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_norm))
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, ce_loss, device: torch.device, positive_class_index: int, threshold: float) -> Tuple[Dict[str, float], pd.DataFrame, float]:
    model.eval()
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    sample_ids: List[str] = []
    patient_ids: List[str] = []
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        rad = batch["rad_features"].to(device)
        labels = batch["label"].to(device)
        outputs = model({"image": images, "rad_features": rad})
        loss = ce_loss(outputs["fusion_logits"], labels)
        total_loss += float(loss.item())
        total_batches += 1

        probs = torch.softmax(outputs["fusion_logits"], dim=1)[:, int(positive_class_index)].detach().cpu().numpy()
        probs_list.append(probs)
        labels_list.append(labels.detach().cpu().numpy().astype(int))
        sample_ids.extend([str(x) for x in batch["sample_id"]])
        patient_ids.extend([str(x) for x in batch["patient_id"]])

    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": (y_prob >= float(threshold)).astype(int),
        }
    )
    return metrics, pred_df, total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    output_dir = args.output_dir or Path(get_required(cfg, "paths.model_selection_output_dir"))
    ensure_dir(output_dir)
    logger = configure_logging(output_dir / "train_preliminary_multimodal_mamba_fusion.log")

    seed = int(get_required(cfg, "cross_validation.seed"))
    set_global_seed(seed)
    device = select_device(args.device)
    positive_class_index = int(get_optional(cfg, "training.positive_class_index", default=1))
    n_splits = int(get_required(cfg, "cross_validation.n_splits"))
    max_epochs = int(get_required(cfg, "training.max_epochs"))
    selection_metric = str(get_optional(cfg, "training.model_selection_metric", default="auc"))
    fixed_threshold = float(get_optional(cfg, "training.fixed_threshold", default=0.5))

    logger.info("Using device: %s", device)

    fold_metric_rows: List[Dict[str, Any]] = []
    oof_rows: List[pd.DataFrame] = []
    best_epochs: List[int] = []

    for outer_fold in range(1, n_splits + 1):
        fold_dir = ensure_dir(output_dir / f"fold_{outer_fold:02d}")
        loaders, fold_meta = build_cv_dataloaders(cfg, outer_fold=outer_fold)
        model = build_model(cfg, radiomics_input_dim=int(fold_meta["radiomics_input_dim"]), device=device)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer, max_epochs=max_epochs)

        cls_weights = compute_class_weights(loaders["train"], num_classes=int(get_required(cfg, "model.num_classes"))).to(device)
        label_smoothing = float(get_optional(cfg, "training.label_smoothing", default=0.05))
        ce_loss = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)

        best_epoch = 0
        best_score = float("-inf")
        best_val_metrics: Optional[Dict[str, float]] = None
        best_val_loss: Optional[float] = None
        epoch_rows: List[Dict[str, Any]] = []

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, ce_loss, device, cfg)
            val_metrics, val_pred_df, val_loss = evaluate_model(
                model,
                loaders["val"],
                ce_loss,
                device,
                positive_class_index=positive_class_index,
                threshold=fixed_threshold,
            )
            score = _selection_score(selection_metric, val_metrics, val_loss)

            epoch_row = {
                "outer_fold": int(outer_fold),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "selection_score": float(score),
            }
            epoch_row.update({f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float, np.floating, np.integer))})
            epoch_rows.append(epoch_row)

            if score > best_score:
                best_score = float(score)
                best_epoch = int(epoch)
                best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
                best_val_loss = float(val_loss)
                torch.save(model.state_dict(), fold_dir / "best_model_state_dict.pth")
                val_pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)

            if scheduler is not None:
                scheduler.step()

        pd.DataFrame(epoch_rows).to_csv(fold_dir / "epoch_metrics.csv", index=False)

        model.load_state_dict(torch.load(fold_dir / "best_model_state_dict.pth", map_location=device))
        best_val_metrics_checked, best_val_pred_df, best_val_loss_checked = evaluate_model(
            model,
            loaders["val"],
            ce_loss,
            device,
            positive_class_index=positive_class_index,
            threshold=fixed_threshold,
        )
        best_val_pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
        oof_rows.append(best_val_pred_df.assign(outer_fold=int(outer_fold)))
        best_epochs.append(int(best_epoch))

        fold_row = {"outer_fold": int(outer_fold), "best_epoch": int(best_epoch), "selection_score": float(best_score), "val_loss": float(best_val_loss_checked)}
        fold_row.update({f"val_{k}": float(v) for k, v in best_val_metrics_checked.items()})
        fold_metric_rows.append(fold_row)

        save_json(
            {
                "outer_fold": int(outer_fold),
                "fold_metadata": fold_meta,
                "best_epoch": int(best_epoch),
                "selection_metric": selection_metric,
                "selection_score": float(best_score),
                "best_validation_metrics": best_val_metrics,
                "best_validation_loss": best_val_loss,
                "fixed_threshold_used_for_fold_evaluation": fixed_threshold,
            },
            fold_dir / "fold_summary.json",
        )
        logger.info("Outer fold %d completed. Best epoch=%d | selection score=%.6f | val AUC=%.6f", outer_fold, best_epoch, best_score, best_val_metrics_checked["auc"])

    cv_metrics_df = pd.DataFrame(fold_metric_rows)
    cv_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    oof_predictions = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)

    recommended_epochs = median_epoch_recommendation(best_epochs)
    selection_summary = {
        "pipeline_role": "optional_preliminary_tuning_only",
        "fold_epoch_summary": summarise_numeric(best_epochs),
        "oof_summary": {
            "n_predictions": int(len(oof_predictions)),
            "pooled_metrics_at_threshold_0_5": compute_binary_metrics(
                oof_predictions["y_true"].to_numpy(dtype=int),
                oof_predictions["y_prob"].to_numpy(dtype=float),
                threshold=fixed_threshold,
            ),
        },
        "final_protocol_recommendation": {
            "recommended_final_training_epochs": int(recommended_epochs),
            "fixed_decision_threshold": float(fixed_threshold),
            "note": "Cross-validation here is for preliminary hyperparameter/epoch selection only. Final reported metrics should come from the held-out final scripts.",
        },
        "environment": collect_environment_metadata(),
    }
    save_json(selection_summary, output_dir / "model_selection_summary.json")
    logger.info("Preliminary model selection completed. Recommended final-training epochs=%d | fixed threshold=%.6f", recommended_epochs, fixed_threshold)


if __name__ == "__main__":
    main()
