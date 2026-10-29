"""Phase 4: Controlled Training Pipeline Optimization Experiments.

Runs isolated, reproducible experiments strictly on training and validation splits:
- Exp 0: Baseline reproduction (pos_weight=5.0, bce=0.5, dice=0.5)
- Exp 1: Reduced positive weight (pos_weight=2.5) to penalize false alarms in rural/agricultural parcels
- Exp 2: Soft Dice dominant loss (bce=0.3, dice=0.7, pos_weight=3.0) for geometric shape coherence
- Exp 3: Dense spatial patch sampling (stride=160) for expanded negative scene diversity

Stores all experimental checkpoints separately in: artifacts/experiments/
Logs metrics to: reports/optimization/experiment_summary.csv
Does NOT touch test scenes during optimization.
"""

import os
import sys
import time
import json
import csv
import copy
from typing import Dict, List, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from models.siam_unet import FCSiamDiff
from models.losses import CombinedBceDiceLoss
from utils.metrics import ChangeDetectionMetrics
from training.validate import validate_epoch
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits


def run_experiment(
    exp_name: str,
    description: str,
    pos_weight: float = 5.0,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    stride: int = 224,
    epochs: int = 15,
    patience: int = 6,
    batch_size: int = 4,
    lr: float = 0.0001,
    seed: int = 42,
    device_str: str = "cpu"
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Description: {description}")
    print(f"Hyperparameters: pos_weight={pos_weight}, bce={bce_weight}, dice={dice_weight}, stride={stride}, lr={lr}")
    print("=" * 70)

    # Determinism
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)

    # Splits
    splits = create_geographic_splits("configs/geographic_splits.json", seed=seed)
    train_scenes = splits["train"]
    val_scenes = splits["val"]

    # Datasets
    train_ds = ChangeDetectionDataset(
        parquet_path="dataset/train-00000-of-00001.parquet",
        split_scenes=train_scenes,
        is_train_split=True,
        patch_size=256,
        stride=stride,
        augment=True
    )
    val_ds = ChangeDetectionDataset(
        parquet_path="dataset/train-00000-of-00001.parquet",
        split_scenes=val_scenes,
        is_train_split=True,
        patch_size=256,
        stride=224,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Patches: {len(train_ds)} | Val Patches: {len(val_ds)}")

    # Model
    model = FCSiamDiff(in_channels=3, base_channels=32, num_classes=1).to(device)

    # Loss & Optimizer
    criterion = CombinedBceDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = 0.0
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    exp_dir = os.path.join("artifacts", "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_save_path = os.path.join(exp_dir, "best_model.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        for batch in train_loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            vmask = batch.get("valid_mask")
            if vmask is not None:
                vmask = vmask.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(t1, t2)
                loss = criterion(logits, mask, valid_mask=vmask)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, batches)

        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, threshold=0.50)
        vf1 = val_metrics["f1"]
        viou = val_metrics["iou"]

        is_best = vf1 > best_val_f1
        if is_best:
            best_val_f1 = vf1
            best_val_iou = viou
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "best_val_iou": best_val_iou,
                "config": {
                    "exp_name": exp_name,
                    "pos_weight": pos_weight,
                    "bce_weight": bce_weight,
                    "dice_weight": dice_weight,
                    "stride": stride,
                    "lr": lr
                }
            }, ckpt_save_path)
        else:
            patience_counter += 1

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Val F1: {vf1:.4f} | Val IoU: {viou:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}" + (" [*BEST*]" if is_best else ""))

        if patience_counter >= patience:
            print(f"  [!] Early stopping at epoch {epoch}")
            break

    total_duration = round(time.time() - start_time, 2)
    print(f"Experiment {exp_name} Finished. Best Epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f} | Best Val IoU: {best_val_iou:.4f} ({total_duration}s)")

    result = {
        "experiment_name": exp_name,
        "description": description,
        "pos_weight": pos_weight,
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
        "stride": stride,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_iou": round(best_val_iou, 4),
        "duration_sec": total_duration,
        "checkpoint_path": ckpt_save_path
    }
    return result


def run_all_experiments():
    os.makedirs("reports/optimization", exist_ok=True)
    summary_csv = "reports/optimization/experiment_summary.csv"

    # Auto-detect device
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running experiments on: {device_str}")

    experiments = [
        {
            "exp_name": "exp0_baseline",
            "description": "Baseline reference (pos_weight=5.0, bce=0.5, dice=0.5, stride=224)",
            "pos_weight": 5.0,
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "stride": 224,
            "epochs": 12,
            "patience": 6
        },
        {
            "exp_name": "exp1_balanced_posweight",
            "description": "Reduced positive weighting (pos_weight=2.5) to penalize false alarms in rural parcels",
            "pos_weight": 2.5,
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "stride": 224,
            "epochs": 12,
            "patience": 6
        },
        {
            "exp_name": "exp2_dice_dominant",
            "description": "Soft Dice dominant loss (bce=0.3, dice=0.7, pos_weight=3.0) for shape coherence",
            "pos_weight": 3.0,
            "bce_weight": 0.3,
            "dice_weight": 0.7,
            "stride": 224,
            "epochs": 12,
            "patience": 6
        },
        {
            "exp_name": "exp3_dense_stride",
            "description": "Dense patch sampling (stride=160, pos_weight=2.5) for expanded negative diversity",
            "pos_weight": 2.5,
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "stride": 160,
            "epochs": 12,
            "patience": 6
        }
    ]

    all_results = []
    for exp in experiments:
        res = run_experiment(
            exp_name=exp["exp_name"],
            description=exp["description"],
            pos_weight=exp["pos_weight"],
            bce_weight=exp["bce_weight"],
            dice_weight=exp["dice_weight"],
            stride=exp["stride"],
            epochs=exp["epochs"],
            patience=exp["patience"],
            device_str=device_str
        )
        all_results.append(res)

    # Save to CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("ALL CONTROLLED EXPERIMENTS COMPLETED")
    print(f"Summary saved to: {summary_csv}")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r['experiment_name']}] Val F1: {r['best_val_f1']:.4f} | Val IoU: {r['best_val_iou']:.4f} | Duration: {r['duration_sec']}s")


if __name__ == "__main__":
    run_all_experiments()
