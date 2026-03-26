"""
Final manuscript-aligned training and held-out evaluation script.

This is the public end-to-end trainer for the multimodal radiomics-3D Mamba
fusion model. Comparator architectures are released separately as model
definitions and are not invoked here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from data.data_loading_final_multimodal import build_final_dataloaders
from utils.analysis_utils import (
    collect_environment_metadata,
    compute_binary_metrics,
    configure_logging,
    ensure_dir,
    get_optional,
    get_required,
    load_json,
    save_json,
    select_device,
    set_global_seed,
)
from models.model_multimodal_mamba_fusion import Mamba_Fusion_Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final manuscript-aligned multimodal training and held-out evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the study configuration JSON file.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Optional override for the output directory.")
    parser.add_argument("--device", type=str, default=None, help="Optional explicit device string, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def build_model(cfg: Mapping[str, Any], radiomics_input_dim: int, device: torch.device) -> nn.Module:
    model = Mamba_Fusion_Model(
        mamba_embed_dim=int(get_required(cfg, "model.mamba_embed_dim")),
        mamba_d_state=int(get_required(cfg, "model.mamba_d_state")),
        mamba_d_conv=int(get_required(cfg, "model.mamba_d_conv")),
        mamba_expand=int(get_required(cfg, "model.mamba_expand")),
        image_in_channels=int(get_required(cfg, "model.image_in_channels")),
        rad_input_features=int(radiomics_input_dim),
        rad_hidden_features=int(get_required(cfg, "model.radiomics_hidden_dim")),
        fusion_hidden_features=int(get_required(cfg, "model.fusion_hidden_dim")),
        num_classes=int(get_required(cfg, "model.num_classes")),
        dropout_rate=float(get_required(cfg, "model.dropout_rate")),
        use_aux_cls=bool(get_optional(cfg, "model.use_auxiliary_image_head", default=True)),
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
    if scheduler_name in {"", "none"}:
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
    fusion_logits = outputs.get("fusion_logits", outputs.get("fusion"))
    if fusion_logits is None:
        raise KeyError("Training outputs must contain 'fusion' or 'fusion_logits'.")
    loss = ce_loss(fusion_logits, labels)

    aux_cfg = get_optional(cfg, "training.auxiliary_loss", default={}) or {}
    if bool(aux_cfg.get("enable", True)):
        image_weight = float(aux_cfg.get("image_weight", 0.0))
        radiomics_weight = float(aux_cfg.get("radiomics_weight", 0.0))
        image_logits = outputs.get("image_logits", outputs.get("img"))
        radiomics_logits = outputs.get("radiomics_logits", outputs.get("rad"))
        if image_logits is not None and image_weight > 0.0:
            loss = loss + image_weight * ce_loss(image_logits, labels)
        if radiomics_logits is not None and radiomics_weight > 0.0:
            loss = loss + radiomics_weight * ce_loss(radiomics_logits, labels)
    return loss


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


def _extract_eval_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, Mapping):
        logits = outputs.get("fusion_logits", outputs.get("fusion"))
        if logits is None:
            raise KeyError("Evaluation outputs must contain 'fusion' or 'fusion_logits'.")
        return logits
    raise TypeError("Unsupported model output type during evaluation.")


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float,
    positive_class_index: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    sample_ids: List[str] = []
    patient_ids: List[str] = []
    cohorts: List[str] = []

    for batch in loader:
        images = batch["image"].to(device)
        rad = batch["rad_features"].to(device)
        labels = batch["label"].to(device)

        outputs = model({"image": images, "rad_features": rad})
        logits = _extract_eval_logits(outputs)
        probs = torch.softmax(logits, dim=1)[:, int(positive_class_index)].detach().cpu().numpy()

        probs_list.append(probs)
        labels_list.append(labels.detach().cpu().numpy().astype(int))
        sample_ids.extend([str(x) for x in batch["sample_id"]])
        patient_ids.extend([str(x) for x in batch["patient_id"]])
        cohorts.extend([str(x) for x in batch["cohort"]])

    if not probs_list:
        raise RuntimeError("Cannot evaluate an empty dataloader.")

    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
    pred_df = pd.DataFrame(
        {
            "cohort": cohorts,
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": (y_prob >= float(threshold)).astype(int),
        }
    )
    return metrics, pred_df


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    default_output_dir = Path(
        get_optional(cfg, "paths.final_output_dir", default=Path(get_required(cfg, "paths.preprocessed_root")) / "final_training")
    )
    output_dir = args.output_dir or default_output_dir
    ensure_dir(output_dir)

    logger = configure_logging(output_dir / "train_final_multimodal_mamba_fusion.log")
    seed = int(get_required(cfg, "cross_validation.seed"))
    set_global_seed(seed)
    device = select_device(args.device)
    positive_class_index = int(get_optional(cfg, "training.positive_class_index", default=1))
    max_epochs = int(get_required(cfg, "training.max_epochs"))
    threshold = float(get_optional(cfg, "training.fixed_threshold", default=0.5))

    logger.info("Using device: %s", device)
    loaders, data_meta = build_final_dataloaders(cfg)
    radiomics_input_dim = int(data_meta["radiomics_input_dim"])
    expected_dim = get_optional(cfg, "model.radiomics_input_dim", default=None)
    if expected_dim is not None and int(expected_dim) != radiomics_input_dim:
        logger.warning(
            "Config radiomics_input_dim=%s differs from selected-table width=%d. Using table width.",
            expected_dim,
            radiomics_input_dim,
        )

    model = build_model(cfg, radiomics_input_dim=radiomics_input_dim, device=device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, max_epochs=max_epochs)
    class_weights = compute_class_weights(loaders["train"], num_classes=int(get_required(cfg, "model.num_classes"))).to(device)
    label_smoothing = float(get_optional(cfg, "training.label_smoothing", default=0.0))
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    logger.info(
        "Starting final training | epochs=%d threshold=%.3f train_n=%d internal_n=%d external_n=%d",
        max_epochs,
        threshold,
        data_meta["development_n"],
        data_meta["internal_test_n"],
        data_meta["external_test_n"],
    )

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, ce_loss, device, cfg)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_rows.append({"epoch": int(epoch), "train_loss": float(train_loss), "learning_rate": current_lr})
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            logger.info("Epoch %03d/%03d | lr=%.6g | train_loss=%.6f", epoch, max_epochs, current_lr, train_loss)

    model_path = output_dir / "final_mamba_fusion.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved final model weights to %s", model_path)

    metric_rows: List[Dict[str, Any]] = []
    prediction_tables: List[pd.DataFrame] = []
    for split_name, loader_key in [
        ("development", "train"),
        ("internal_test", "internal_test"),
        ("external_test", "external_test"),
    ]:
        loader = loaders[loader_key]
        if len(loader.dataset) == 0:
            logger.warning("Skipping empty split: %s", split_name)
            continue
        metrics, pred_df = evaluate_split(
            model,
            loader,
            device=device,
            threshold=threshold,
            positive_class_index=positive_class_index,
        )
        metric_rows.append({"split": split_name, **metrics})
        prediction_tables.append(pred_df.assign(split=split_name))
        logger.info(
            "[%s] auc=%.4f acc=%.4f sens=%.4f spec=%.4f n=%d",
            split_name,
            metrics["auc"],
            metrics["accuracy"],
            metrics["sensitivity"],
            metrics["specificity"],
            metrics["n_samples"],
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "final_evaluation_metrics.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output_dir / "training_history.csv", index=False)
    if prediction_tables:
        pd.concat(prediction_tables, ignore_index=True).to_csv(output_dir / "final_predictions.csv", index=False)

    save_json(
        {
            "config_path": str(args.config),
            "output_dir": str(output_dir),
            "device": str(device),
            "seed": int(seed),
            "radiomics_input_dim": int(radiomics_input_dim),
            "data_summary": data_meta,
            "environment": collect_environment_metadata(),
            "final_metrics": metric_rows,
        },
        output_dir / "final_run_summary.json",
    )
    logger.info("Final training and held-out evaluation completed successfully.")


if __name__ == "__main__":
    main()
