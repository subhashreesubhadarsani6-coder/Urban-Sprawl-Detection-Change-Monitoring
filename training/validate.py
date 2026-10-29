"""Validation Module for Siamese U-Net Change Detection.

Evaluates model over validation or test DataLoader, computing loss and global
confusion matrix metrics (Precision, Recall, F1, IoU) strictly accumulated across
all valid pixels without batch-averaging bias.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import ChangeDetectionMetrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """Run validation loop over loader and compute accumulated metrics.

    Args:
        model: FC-Siam-diff model instance.
        val_loader: Validation DataLoader.
        criterion: Combined loss function.
        device: Active torch device (cuda or cpu).
        threshold: Classification probability threshold.

    Returns:
        (avg_val_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metrics_tracker = ChangeDetectionMetrics()

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for batch in val_loader:
                t1 = batch["image_t1"].to(device, non_blocking=True)
                t2 = batch["image_t2"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(device, non_blocking=True)

                logits = model(t1, t2)
                loss = criterion(logits, mask, valid_mask=valid_mask)

                total_loss += loss.item()
                num_batches += 1

                # Accumulate global confusion matrix
                metrics_tracker.update(
                    predictions=logits,
                    targets=mask,
                    valid_mask=valid_mask,
                    threshold=threshold
                )

    avg_loss = total_loss / max(1, num_batches)
    metrics = metrics_tracker.compute()
    metrics["loss"] = float(round(avg_loss, 5))
    return avg_loss, metrics
