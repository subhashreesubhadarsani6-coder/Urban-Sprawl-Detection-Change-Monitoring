"""Evaluation Script for Held-Out Test Scenes.

Loads verified trained checkpoint, applies selected validation threshold,
and evaluates global and per-scene metrics across the 10 held-out OSCD test scenes.
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits
from src.inference.predictor import ChangePredictor
from src.metrics import ChangeDetectionMetrics

def evaluate_test_set(
    checkpoint_path: str = "checkpoints/best_checkpoint.pth",
    config_path: str = "configs/train.yaml",
    threshold: float = None,
    output_report: str = "reports/evaluation_report.json"
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_file = cfg["dataset"].get("splits_file", "configs/geographic_splits.json")
    splits = create_geographic_splits(output_path=splits_file)
    test_scenes = splits["test"]

    print(f"Loading checkpoint from: {checkpoint_path}")
    predictor = ChangePredictor(checkpoint_path=checkpoint_path, threshold=threshold)
    eval_threshold = predictor.threshold
    print(f"Evaluation Threshold: {eval_threshold}")

    # Load test scenes dataset (full-scene mode)
    test_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["test_parquet"],
        split_scenes=test_scenes,
        is_train_split=False,
        patch_size=None,  # Full-scene mode
        augment=False
    )

    global_tracker = ChangeDetectionMetrics()
    per_scene_metrics = {}

    print(f"\n--- Evaluating {len(test_scenes)} Held-Out Geographic Scenes ---")
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        scene_id = item["scene_id"]
        t1_np = item["image_t1"].numpy().transpose(1, 2, 0)
        t2_np = item["image_t2"].numpy().transpose(1, 2, 0)
        gt_mask = item["mask"].numpy().squeeze().astype(np.uint8)
        valid_mask = item["valid_mask"].numpy().squeeze().astype(np.uint8)

        # Denormalize to uint8 for sliding window tiled predictor
        t1_u8 = np.clip((t1_np * 0.224 + 0.45) * 255.0, 0, 255).astype(np.uint8)
        t2_u8 = np.clip((t2_np * 0.224 + 0.45) * 255.0, 0, 255).astype(np.uint8)

        pred_res = predictor.predict_scene(t1_u8, t2_u8, patch_size=256, overlap=64)
        pred_binary = pred_res["binary_mask"]

        # Track per-scene metrics
        scene_tracker = ChangeDetectionMetrics()
        scene_tracker.update(pred_binary, gt_mask, valid_mask=valid_mask, threshold=0.5)
        m = scene_tracker.compute()
        per_scene_metrics[scene_id] = m

        # Accumulate to global tracker
        global_tracker.update(pred_binary, gt_mask, valid_mask=valid_mask, threshold=0.5)

        print(
            f"  Scene [{idx+1:02d}/{len(test_scenes):02d}] {scene_id:12s}: "
            f"F1={m['f1']:.4f} | IoU={m['iou']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | "
            f"Changed Px={m['tp'] + m['fn']:,}"
        )

    global_metrics = global_tracker.compute()
    print("\n=== Global Held-Out Test Results ===")
    print(f"Threshold:    {eval_threshold}")
    print(f"F1 Score:     {global_metrics['f1']:.4f}")
    print(f"IoU:          {global_metrics['iou']:.4f}")
    print(f"Precision:    {global_metrics['precision']:.4f}")
    print(f"Recall:       {global_metrics['recall']:.4f}")
    print(f"Pixel Acc:    {global_metrics['accuracy']:.4f}")
    print(f"Valid Pixels: {global_metrics['valid_pixels']:,}")

    report = {
        "checkpoint": checkpoint_path,
        "threshold": eval_threshold,
        "global_metrics": global_metrics,
        "per_scene_metrics": per_scene_metrics,
        "scenes_evaluated": test_scenes
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation report saved to: {output_report}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate FC-Siam-diff model on test set")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pth")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
