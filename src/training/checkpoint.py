"""Checkpoint Management for Siamese U-Net Change Detection.

Handles atomic saving, loading, validation, and registration of training checkpoints,
preserving complete optimizer, scheduler, AMP scaler states, metrics, and architecture metadata.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    threshold: Optional[float] = None
):
    """Atomically save checkpoint with comprehensive metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "in_channels": getattr(model, "in_channels", 3),
        "base_channels": getattr(model, "base_channels", 32),
        "num_classes": getattr(model, "num_classes", 1),
        "threshold": threshold
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, temp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load and validate checkpoint into model and training components.

    Returns:
        Checkpoint dictionary with metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = device if device else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Validate model parameters compatibility
    expected_in = getattr(model, "in_channels", None)
    saved_in = checkpoint.get("in_channels")
    if expected_in and saved_in and expected_in != saved_in:
        raise ValueError(f"Incompatible channel count: Model expects {expected_in}, checkpoint has {saved_in}")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
