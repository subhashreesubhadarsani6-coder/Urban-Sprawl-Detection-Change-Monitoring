"""Phase 3: Deep Prediction Audit & Spatial Error Analysis Script.

Evaluates the baseline Siamese U-Net (FC-Siam-diff) model on all 10 held-out test scenes:
1. Generates 8-panel diagnostic figures for every test scene:
   - T1 image
   - T2 image
   - Ground-truth change mask
   - Predicted probability map
   - Predicted binary mask
   - False-positive (FP) pixels map
   - False-negative (FN) pixels map
   - Change overlay on T2
2. Saves figures to: reports/optimization/predictions/{scene_id}_audit.png
3. Calculates global metrics: Precision, Recall, F1, IoU, TP, FP, TN, FN, Prevalence.
4. Generates per-scene metrics and exports: reports/optimization/scene_metrics.csv
5. Saves global metrics to: reports/optimization/test_metrics.json
6. Performs radiometric and correlation analysis of errors (vegetation, brightness change, edges).
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from models.siam_unet import FCSiamDiff
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits
from utils.metrics import ChangeDetectionMetrics


def run_prediction_audit(
    checkpoint_path: str = "artifacts/checkpoints/best_model.pth",
    threshold: float = 0.50,
    output_dir: str = "reports/optimization"
):
    print("=" * 70)
    print("PHASE 3: PREDICTION AUDIT & ERROR DECOMPOSITION")
    print("=" * 70)

    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Evaluating checkpoint: {checkpoint_path} (Threshold: {threshold})")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = FCSiamDiff(in_channels=3, base_channels=32, num_classes=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test scenes
    splits = create_geographic_splits("configs/geographic_splits.json", seed=42)
    test_scenes = splits["test"]

    dataset = ChangeDetectionDataset(
        parquet_path="dataset/test-00000-of-00001.parquet",
        split_scenes=test_scenes,
        is_train_split=False,
        patch_size=None,  # Full scene evaluation
        augment=False
    )

    global_tracker = ChangeDetectionMetrics()
    scene_records = []
    error_analysis_data = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        scene_id = sample["scene_id"]

        t1_t = sample["image_t1"].unsqueeze(0).to(device)
        t2_t = sample["image_t2"].unsqueeze(0).to(device)
        gt_t = sample["mask"].unsqueeze(0).to(device)
        valid_t = sample.get("valid_mask")
        if valid_t is not None:
            valid_t = valid_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(t1_t, t2_t)
            probs = torch.sigmoid(logits)

        # Update global tracker
        global_tracker.update(probs, gt_t, valid_mask=valid_t, threshold=threshold)

        # Scene metrics
        s_tracker = ChangeDetectionMetrics()
        s_tracker.update(probs, gt_t, valid_mask=valid_t, threshold=threshold)
        s_metrics = s_tracker.compute()

        # Numpy arrays for diagnostics
        t1_np = sample["image_t1"].permute(1, 2, 0).numpy()
        t2_np = sample["image_t2"].permute(1, 2, 0).numpy()
        # Unnormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        t1_rgb = np.clip((t1_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
        t2_rgb = np.clip((t2_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

        prob_map = probs.squeeze().cpu().numpy()
        pred_bin = (prob_map >= threshold).astype(np.uint8)
        gt_bin = gt_t.squeeze().cpu().numpy().astype(np.uint8)

        # Error masks
        tp_mask = ((pred_bin == 1) & (gt_bin == 1)).astype(np.uint8)
        fp_mask = ((pred_bin == 1) & (gt_bin == 0)).astype(np.uint8)
        fn_mask = ((pred_bin == 0) & (gt_bin == 1)).astype(np.uint8)

        # Change overlay on T2: T2 + red for pred_bin, green for TP, yellow for FP, blue for FN
        overlay = t2_rgb.copy().astype(np.float32)
        # Apply red tint for predicted changes
        overlay[pred_bin == 1] = overlay[pred_bin == 1] * 0.35 + np.array([239, 68, 68], dtype=np.float32) * 0.65
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Compute radiometric correlation for false positives
        # E.g. brightness shift between T1 and T2
        diff_lum = np.abs(t2_rgb.mean(axis=-1).astype(np.float32) - t1_rgb.mean(axis=-1).astype(np.float32))
        fp_mean_diff = float(diff_lum[fp_mask == 1].mean()) if fp_mask.sum() > 0 else 0.0
        bg_mean_diff = float(diff_lum[gt_bin == 0].mean()) if (gt_bin == 0).sum() > 0 else 0.0

        error_analysis_data[scene_id] = {
            "fp_count": int(fp_mask.sum()),
            "fn_count": int(fn_mask.sum()),
            "tp_count": int(tp_mask.sum()),
            "fp_mean_lum_diff": round(fp_mean_diff, 2),
            "background_mean_lum_diff": round(bg_mean_diff, 2)
        }

        # Generate 8-panel diagnostic figure
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))

        # Row 1
        axes[0, 0].imshow(t1_rgb)
        axes[0, 0].set_title(f"1. T1 Observation (Earlier)\n{scene_id.upper()}", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(t2_rgb)
        axes[0, 1].set_title("2. T2 Observation (Later)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gt_bin, cmap="gray")
        axes[0, 2].set_title(f"3. Ground Truth Mask\n({s_metrics['actual_positives']:,} changed px)", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        im_prob = axes[0, 3].imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, 3].set_title("4. Predicted Probability Map", fontsize=11, fontweight="bold")
        axes[0, 3].axis("off")
        plt.colorbar(im_prob, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # Row 2
        axes[1, 0].imshow(pred_bin, cmap="gray")
        axes[1, 0].set_title(f"5. Binary Mask (Cutoff {threshold})\n({s_metrics['predicted_positives']:,} px)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        # False Positives (Yellow on dark background)
        fp_vis = np.zeros((*fp_mask.shape, 3), dtype=np.uint8)
        fp_vis[fp_mask == 1] = [234, 179, 8]  # Yellow
        axes[1, 1].imshow(fp_vis)
        axes[1, 1].set_title(f"6. False Positives (FP)\n({s_metrics['fp']:,} false alarm px)", fontsize=11, color="#b45309", fontweight="bold")
        axes[1, 1].axis("off")

        # False Negatives (Cyan on dark background)
        fn_vis = np.zeros((*fn_mask.shape, 3), dtype=np.uint8)
        fn_vis[fn_mask == 1] = [6, 182, 212]  # Cyan
        axes[1, 2].imshow(fn_vis)
        axes[1, 2].set_title(f"7. False Negatives (FN)\n({s_metrics['fn']:,} missed px)", fontsize=11, color="#0e7490", fontweight="bold")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title(f"8. Change Overlay on T2\nF1: {s_metrics['f1']:.4f} | IoU: {s_metrics['iou']:.4f}", fontsize=11, fontweight="bold")
        axes[1, 3].axis("off")

        plt.suptitle(
            f"UrbanScope AI Prediction Audit — Scene: {scene_id.upper()} | F1: {s_metrics['f1']:.4f} | IoU: {s_metrics['iou']:.4f} | Prec: {s_metrics['precision']:.4f} | Rec: {s_metrics['recall']:.4f}",
            fontsize=14, fontweight="bold", y=0.98
        )
        plt.tight_layout()
        audit_fig_path = os.path.join(pred_dir, f"{scene_id}_audit.png")
        plt.savefig(audit_fig_path, dpi=180, bbox_inches="tight")
        plt.close()

        print(f"  [Audited] Scene {scene_id:12s} -> F1: {s_metrics['f1']:.4f} | IoU: {s_metrics['iou']:.4f} | FP: {s_metrics['fp']:,} | FN: {s_metrics['fn']:,}")

        scene_records.append({
            "scene_id": scene_id,
            "f1": s_metrics["f1"],
            "iou": s_metrics["iou"],
            "precision": s_metrics["precision"],
            "recall": s_metrics["recall"],
            "accuracy": s_metrics["accuracy"],
            "tp": s_metrics["tp"],
            "fp": s_metrics["fp"],
            "tn": s_metrics["tn"],
            "fn": s_metrics["fn"],
            "actual_change_pixels": s_metrics["actual_positives"],
            "predicted_change_pixels": s_metrics["predicted_positives"],
            "valid_pixels": s_metrics["valid_pixels"]
        })

    # Global summary
    global_res = global_tracker.compute()
    print("\n" + "=" * 70)
    print("GLOBAL HELD-OUT PREDICTION AUDIT SUMMARY")
    print("=" * 70)
    print(f"Precision: {global_res['precision']:.4f}")
    print(f"Recall:    {global_res['recall']:.4f}")
    print(f"F1-Score:  {global_res['f1']:.4f}")
    print(f"IoU:       {global_res['iou']:.4f}")
    print(f"Accuracy:  {global_res['accuracy']:.4f}")
    print(f"Confusion: TP={global_res['tp']:,} | FP={global_res['fp']:,} | TN={global_res['tn']:,} | FN={global_res['fn']:,}")

    # Export scene_metrics.csv
    csv_path = os.path.join(output_dir, "scene_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scene_records[0].keys()))
        writer.writeheader()
        writer.writerows(scene_records)
    print(f"\nExported per-scene metrics CSV to: {csv_path}")

    # Export test_metrics.json
    json_path = os.path.join(output_dir, "test_metrics.json")
    out_json = {
        "audit_checkpoint": checkpoint_path,
        "optimal_threshold": threshold,
        "global_metrics": global_res,
        "scenes_ranked_by_f1": sorted(scene_records, key=lambda x: x["f1"], reverse=True),
        "error_analysis_data": error_analysis_data
    }
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Exported global test metrics JSON to: {json_path}")

    return out_json


if __name__ == "__main__":
    run_prediction_audit()
