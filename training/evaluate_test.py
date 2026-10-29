"""Phase I: Final Held-Out Test Evaluation Pipeline.

Evaluates the trained Siamese U-Net (FC-Siam-diff) checkpoint on held-out test scenes:
1. Loads best checkpoint from artifacts/checkpoints/best_model.pth.
2. Tunes probability threshold strictly on the validation split.
3. Evaluates selected threshold on the 10 held-out test scenes without label leakage.
4. Computes global metrics: Precision, Recall, F1-score, IoU, TP, FP, TN, FN.
5. Computes per-scene metrics across each independent test metropolitan region.
6. Generates visual test overlays and saves reports/test_metrics.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Ensure project root in python path
sys.path.insert(0, os.path.abspath("."))

from models.siam_unet import FCSiamDiff
from utils.metrics import ChangeDetectionMetrics
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits
from training.visualize import save_visual_comparison


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: List[float] = None
) -> float:
    """Tune probability threshold using validation data only."""
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    model.eval()
    all_logits = []
    all_masks = []
    all_valids = []

    with torch.no_grad():
        for batch in val_loader:
            t1 = batch["image_t1"].to(device)
            t2 = batch["image_t2"].to(device)
            logits = model(t1, t2)
            all_logits.append(logits.cpu())
            all_masks.append(batch["mask"])
            all_valids.append(batch.get("valid_mask"))

    best_thresh = 0.5
    best_f1 = -1.0

    print("--- Validation Threshold Calibration ---")
    for thresh in thresholds:
        tracker = ChangeDetectionMetrics()
        for l, m, v in zip(all_logits, all_masks, all_valids):
            tracker.update(predictions=l, targets=m, valid_mask=v, threshold=thresh)
        res = tracker.compute()
        print(f"  Threshold {thresh:.2f} -> Val F1: {res['f1']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | IoU: {res['iou']:.4f}")
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_thresh = thresh

    print(f"Selected Optimal Threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
    return best_thresh


def evaluate_test_set(
    checkpoint_path: str = "artifacts/checkpoints/best_model.pth",
    config_path: str = "training/config.yaml",
    output_report: str = "reports/test_metrics.json"
):
    """Run full held-out test evaluation."""
    print("=" * 70)
    print("PHASE I: HELD-OUT TEST EVALUATION")
    print("=" * 70)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Checkpoint
    if not os.path.exists(checkpoint_path):
        # Fallback to checkpoints/best_checkpoint.pth if available
        fallback = "checkpoints/best_checkpoint.pth"
        if os.path.exists(fallback):
            print(f"Notice: {checkpoint_path} not found, falling back to existing {fallback}")
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = FCSiamDiff(
        in_channels=cfg["model"].get("in_channels", 3),
        base_channels=cfg["model"].get("base_channels", 32),
        num_classes=cfg["model"].get("num_classes", 1)
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Load Splits
    splits_file = cfg["dataset"].get("splits_file", "configs/geographic_splits.json")
    splits = create_geographic_splits(output_path=splits_file, seed=cfg["reproducibility"]["seed"])
    val_scenes = splits["val"]
    test_scenes = splits["test"]

    # 1. Validation loader for threshold calibration
    val_dataset = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=val_scenes,
        is_train_split=True,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimal_thresh = find_optimal_threshold(model, val_loader, device)

    # 2. Test scenes evaluation (scene-by-scene full imagery)
    print(f"\nEvaluating on {len(test_scenes)} Held-Out Test Scenes: {test_scenes}...")
    global_tracker = ChangeDetectionMetrics()
    per_scene_metrics = {}

    # Load full scenes from test parquet
    test_dataset = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["test_parquet"],
        split_scenes=test_scenes,
        is_train_split=False,
        patch_size=None,  # Full scene evaluation
        augment=False
    )

    reports_dir = Path("reports/training")
    reports_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        scene_id = sample["scene_id"]

        t1 = sample["image_t1"].unsqueeze(0).to(device)
        t2 = sample["image_t2"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        valid = sample.get("valid_mask")
        if valid is not None:
            valid = valid.unsqueeze(0).to(device)

        # Sliding window or direct inference for scene
        with torch.no_grad():
            # If scene fits in memory, direct forward pass
            logits = model(t1, t2)
            probs = torch.sigmoid(logits)

        # Scene tracker
        scene_tracker = ChangeDetectionMetrics()
        scene_tracker.update(predictions=probs, targets=mask, valid_mask=valid, threshold=optimal_thresh)
        s_res = scene_tracker.compute()
        per_scene_metrics[scene_id] = s_res

        # Update global tracker
        global_tracker.update(predictions=probs, targets=mask, valid_mask=valid, threshold=optimal_thresh)

        print(f"  Scene [{scene_id:12s}] -> F1: {s_res['f1']:.4f} | IoU: {s_res['iou']:.4f} | Changed Pixels: {s_res['actual_positives']:,}")

        # Save sample overlay for representative scenes (e.g. lasvegas)
        if scene_id in ["lasvegas", "dubai", "milano"]:
            prob_np = probs.squeeze().cpu().numpy()
            pred_mask_np = (prob_np >= optimal_thresh).astype(np.uint8)
            gt_np = mask.squeeze().cpu().numpy()

            mean = np.array(cfg["dataset"]["normalization"]["mean"])
            std = np.array(cfg["dataset"]["normalization"]["std"])
            t1_np = (t1.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0
            t2_np = (t2.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0

            vis_path = reports_dir / f"test_evaluation_{scene_id}.png"
            save_visual_comparison(
                t1_img=t1_np.clip(0, 255),
                t2_img=t2_np.clip(0, 255),
                gt_mask=gt_np,
                pred_prob=prob_np,
                pred_mask=pred_mask_np,
                scene_id=scene_id,
                output_path=str(vis_path)
            )

    global_res = global_tracker.compute()
    print("\n" + "=" * 70)
    print("GLOBAL TEST SET METRICS (Held-out Scenes)")
    print("=" * 70)
    print(f"Optimal Probability Cutoff: {optimal_thresh:.2f}")
    print(f"Test Precision:  {global_res['precision']:.4f}")
    print(f"Test Recall:     {global_res['recall']:.4f}")
    print(f"Test F1-Score:   {global_res['f1']:.4f}")
    print(f"Test IoU:        {global_res['iou']:.4f}")
    print(f"Test Accuracy:   {global_res['accuracy']:.4f}")
    print(f"Confusion Matrix: TP={global_res['tp']:,} | FP={global_res['fp']:,} | TN={global_res['tn']:,} | FN={global_res['fn']:,}")

    # Compile Final JSON Report
    report = {
        "status": "COMPLETED",
        "checkpoint_evaluated": checkpoint_path,
        "optimal_threshold_tuned_on_val": optimal_thresh,
        "global_test_metrics": global_res,
        "per_scene_metrics": per_scene_metrics,
        "num_test_scenes": len(test_scenes),
        "test_scenes": test_scenes
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal test evaluation report saved to: {output_report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Siamese U-Net on held-out test scenes.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--output", type=str, default="reports/test_metrics.json")
    args = parser.parse_args()

    evaluate_test_set(checkpoint_path=args.checkpoint, config_path=args.config, output_report=args.output)
