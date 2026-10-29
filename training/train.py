"""Production Training Pipeline for Siamese U-Net Change Detection (FC-Siam-diff).

Executes complete training loop adhering to all Phase E, F, G, H specifications:
1. Environment and device configuration (CUDA if available, CPU fallback).
2. Reproducibility seed setup.
3. Patch-level dataset construction from geographic splits without spatial leakage.
4. FC-Siam-diff architecture with AdamW and CosineAnnealingLR.
5. Combined BCEWithLogitsLoss + Soft Dice Loss with nodata masking.
6. Epoch-by-epoch training and validation with global metric accumulation.
7. Early stopping and checkpoint persistence in artifacts/checkpoints/.
8. Training history CSV, config JSON, and split manifest serialization.
9. Comprehensive learning curves and visualization artifacts in reports/training/.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from models.siam_unet import FCSiamDiff
from models.losses import CombinedBceDiceLoss
from utils.metrics import ChangeDetectionMetrics
from training.validate import validate_epoch
from training.visualize import generate_training_curves, save_visual_comparison
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits


def set_seed(seed: int = 42):
    """Set random seed across all libraries for deterministic execution."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str = "artifacts/checkpoints",
    best_filename: str = "best_model.pth",
    latest_filename: str = "last_checkpoint.pth"
):
    """Save latest recovery checkpoint and best-performing model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, latest_filename)
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        torch.save(state, best_path)
        print(f"  [+] Saved new best model checkpoint to: {best_path} (Val F1: {state['best_val_f1']:.4f})")


def run_training(config_path: str = "training/config.yaml", resume: bool = False, smoke_test: bool = False):
    """Main training routine."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Device and Environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("SIAMESE U-NET (FC-SIAM-DIFF) CHANGE DETECTION TRAINING PIPELINE")
    print("=" * 70)
    print(f"Active Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Running on CPU (CUDA not active in PyTorch environment).")

    # 2. Reproducibility
    seed = cfg["reproducibility"].get("seed", 42)
    set_seed(seed)
    print(f"Random Seed:   {seed} (deterministic mode enabled)")

    # 3. Geographic Dataset Splits
    splits_file = cfg["dataset"].get("splits_file", "configs/geographic_splits.json")
    splits = create_geographic_splits(output_path=splits_file, seed=seed)
    train_scenes = splits["train"]
    val_scenes = splits["val"]
    test_scenes = splits["test"]

    print(f"\n--- Geographic Splits Manifest ---")
    print(f"Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val Scenes   ({len(val_scenes)}): {val_scenes}")
    print(f"Test Scenes  ({len(test_scenes)}): {test_scenes}")

    # Save split manifest to checkpoints directory as required in Phase G
    save_dir = cfg["checkpoints"].get("save_dir", "artifacts/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "split_manifest.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # 4. Data Loaders
    patch_size = cfg["dataset"].get("patch_size", 256)
    stride = cfg["dataset"].get("stride", 224)
    batch_size = 2 if smoke_test else cfg["training"].get("batch_size", 4)
    num_workers = cfg["training"].get("num_workers", 0)

    print(f"\nConstructing DataLoaders (patch_size={patch_size}, stride={stride}, batch_size={batch_size})...")
    train_dataset = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=train_scenes,
        is_train_split=True,
        patch_size=patch_size,
        stride=stride,
        min_valid_ratio=cfg["dataset"].get("min_valid_ratio", 0.5),
        augment=True,
        mean=tuple(cfg["dataset"]["normalization"]["mean"]),
        std=tuple(cfg["dataset"]["normalization"]["std"])
    )

    val_dataset = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=val_scenes,
        is_train_split=True,
        patch_size=patch_size,
        stride=stride,
        min_valid_ratio=cfg["dataset"].get("min_valid_ratio", 0.5),
        augment=False,
        mean=tuple(cfg["dataset"]["normalization"]["mean"]),
        std=tuple(cfg["dataset"]["normalization"]["std"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"Train Dataset: {len(train_dataset)} patches ({len(train_loader)} batches)")
    print(f"Val Dataset:   {len(val_dataset)} patches ({len(val_loader)} batches)")

    # 5. Model Architecture
    model = FCSiamDiff(
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 32),
        num_classes=cfg["model"].get("num_classes", 1)
    ).to(device)
    print(f"\n{model.summary()}")

    # 6. Loss Function with Class Imbalance Weighting
    loss_cfg = cfg["training"]["loss"]
    criterion = CombinedBceDiceLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        pos_weight=loss_cfg.get("pos_weight", 5.0),
        smooth=loss_cfg.get("smooth", 1.0)
    )

    # 7. Optimizer and Learning Rate Scheduler
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 0.0001)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01))
    )

    total_epochs = 2 if smoke_test else cfg["training"].get("epochs", 40)
    sched_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=float(sched_cfg.get("eta_min", 0.000001))
    )

    # 8. Checkpoint Recovery
    start_epoch = 1
    best_val_f1 = 0.0
    best_val_iou = 0.0
    patience = cfg["training"]["early_stopping"].get("patience", 8)
    patience_counter = 0
    history = []

    latest_ckpt_path = os.path.join(save_dir, cfg["checkpoints"].get("latest_filename", "last_checkpoint.pth"))
    if resume and os.path.exists(latest_ckpt_path):
        print(f"\nResuming training from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        history = checkpoint.get("history", [])
        print(f"Resumed at epoch {start_epoch} (Previous Best Val F1: {best_val_f1:.4f})")

    # Serialize training config
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 9. Training Loop
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING ({total_epochs} epochs | Monitor: Validation F1)")
    print("=" * 70)

    grad_clip = cfg["training"].get("gradient_clip", 1.0)
    best_epoch = start_epoch
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("CUDA Automatic Mixed Precision (AMP) enabled for accelerated Tensor Core throughput.")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(t1, t2)
                loss = criterion(logits, mask, valid_mask=valid_mask)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if smoke_test and batch_idx >= 1:
                break  # Short smoke test

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation Phase
        avg_val_loss, val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=0.5
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_duration = round(time.time() - epoch_start_time, 2)

        # Log Metrics
        val_f1 = val_metrics["f1"]
        val_iou = val_metrics["iou"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "val_iou": val_iou,
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_tn": val_metrics["tn"],
            "val_fn": val_metrics["fn"],
            "lr": current_lr,
            "duration_sec": epoch_duration
        }
        history.append(epoch_record)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | "
            f"Time: {epoch_duration}s"
            + (" [*BEST*]" if is_best else "")
        )

        # Save Checkpoints
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "model_config": cfg["model"],
            "history": history
        }
        save_checkpoint(
            state=ckpt_state,
            is_best=is_best,
            save_dir=save_dir,
            best_filename=cfg["checkpoints"].get("best_filename", "best_model.pth"),
            latest_filename=cfg["checkpoints"].get("latest_filename", "last_checkpoint.pth")
        )

        # Early Stopping
        if patience_counter >= patience and not smoke_test:
            print(f"\n[!] Early stopping triggered: Validation F1 has not improved for {patience} epochs.")
            break

    # 10. Persist Training History CSV
    history_csv_path = os.path.join(save_dir, "training_history.csv")
    if history:
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"\nSaved training history CSV to: {history_csv_path}")

    # 11. Generate Learning Curves and Visualizations
    reports_dir = cfg["reports"].get("dir", "reports/training")
    generate_training_curves(history, output_dir=reports_dir)

    # 12. Generate Sample Prediction Visual Comparison
    print("\nGenerating sample visual predictions on representative scene...")
    model.eval()
    sample_vis_path = os.path.join(reports_dir, "sample_prediction_comparison.png")
    try:
        # Evaluate on the first patch of val dataset
        sample_batch = val_dataset[0]
        s_t1 = sample_batch["image_t1"].unsqueeze(0).to(device)
        s_t2 = sample_batch["image_t2"].unsqueeze(0).to(device)
        with torch.no_grad():
            s_logits = model(s_t1, s_t2)
            s_prob = torch.sigmoid(s_logits).squeeze().cpu().numpy()
            s_pred_mask = (s_prob >= 0.5).astype(np.uint8)

        # Unnormalize imagery for visual comparison
        mean = np.array(cfg["dataset"]["normalization"]["mean"])
        std = np.array(cfg["dataset"]["normalization"]["std"])
        s_t1_np = (s_t1.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0
        s_t2_np = (s_t2.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0
        s_gt_mask = sample_batch["mask"].squeeze().cpu().numpy()

        save_visual_comparison(
            t1_img=s_t1_np.clip(0, 255),
            t2_img=s_t2_np.clip(0, 255),
            gt_mask=s_gt_mask,
            pred_prob=s_prob,
            pred_mask=s_pred_mask,
            scene_id=sample_batch["scene_id"],
            output_path=sample_vis_path
        )
    except Exception as e:
        print(f"Warning: Failed to generate visual sample: {e}")

    # 13. Compile Training Summary JSON
    summary = {
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": cfg["model"]["name"],
        "trainable_parameters": model.count_parameters(),
        "device": str(device),
        "total_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_iou": round(best_val_iou, 4),
        "best_checkpoint_path": os.path.join(save_dir, cfg["checkpoints"].get("best_filename", "best_model.pth")),
        "latest_checkpoint_path": latest_ckpt_path,
        "history_csv_path": history_csv_path,
        "learning_curves_dir": reports_dir,
        "sample_prediction_image": sample_vis_path,
        "geographic_splits": {
            "train_scenes": len(train_scenes),
            "val_scenes": len(val_scenes),
            "test_scenes": len(test_scenes)
        }
    }

    summary_path = cfg["reports"].get("summary_file", "reports/training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print(f"Best Epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f} | Best Val IoU: {best_val_iou:.4f}")
    print(f"Summary JSON saved to: {summary_path}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FC-Siam-diff change detection model.")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke test with 2 batches")
    args = parser.parse_args()

    run_training(config_path=args.config, resume=args.resume, smoke_test=args.smoke_test)
