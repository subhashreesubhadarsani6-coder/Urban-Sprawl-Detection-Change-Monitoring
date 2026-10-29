"""Training Visualization and Reporting Module.

Generates:
1. Training vs Validation Loss Curve
2. Validation F1 Score Curve
3. Validation IoU Curve
4. Precision & Recall Curves
5. Learning Rate Decay Curve
6. Sample T1 / T2 / Ground-Truth / Prediction Visual Comparison
Saves all figures to reports/training/ and compiles reports/training_summary.json.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def generate_training_curves(history: List[Dict[str, Any]], output_dir: str = "reports/training"):
    """Plot and save all learning curves from training history."""
    os.makedirs(output_dir, exist_ok=True)
    if not history:
        print("No training history available to plot.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_iou = [h["val_iou"] for h in history]
    val_prec = [h["val_precision"] for h in history]
    val_rec = [h["val_recall"] for h in history]
    lr = [h["lr"] for h in history]

    if MATPLOTLIB_AVAILABLE:
        # 1. Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss", color="#3b82f6", linewidth=2)
        plt.plot(epochs, val_loss, label="Val Loss", color="#ef4444", linewidth=2)
        plt.title("Training and Validation Loss Curve", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Combined BCE + Dice Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
        plt.close()

        # 2. Validation F1 Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_f1, label="Val F1-Score", color="#10b981", linewidth=2, marker="o")
        plt.title("Validation F1-Score Progression", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_f1_curve.png"), dpi=200)
        plt.close()

        # 3. Validation IoU Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_iou, label="Val IoU", color="#8b5cf6", linewidth=2, marker="s")
        plt.title("Validation IoU Progression", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Intersection-over-Union (IoU)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_iou_curve.png"), dpi=200)
        plt.close()

        # 4. Precision and Recall Curves
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_prec, label="Precision", color="#06b6d4", linewidth=2)
        plt.plot(epochs, val_rec, label="Recall", color="#f59e0b", linewidth=2)
        plt.title("Validation Precision vs Recall", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Metric Score", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=200)
        plt.close()

        # 5. Learning Rate Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, lr, label="Learning Rate", color="#64748b", linewidth=2)
        plt.title("Cosine Annealing Learning Rate Schedule", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_schedule_curve.png"), dpi=200)
        plt.close()

        print(f"Generated 5 learning curve plots in {output_dir}/")


def save_visual_comparison(
    t1_img: np.ndarray,
    t2_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    scene_id: str,
    output_path: str
):
    """Generate side-by-side 5-panel visual comparison: [T1, T2, GT Mask, Pred Prob, Pred Mask]."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize uint8
    t1 = t1_img.astype(np.uint8) if t1_img.dtype != np.uint8 else t1_img
    t2 = t2_img.astype(np.uint8) if t2_img.dtype != np.uint8 else t2_img

    gt_vis = (gt_mask * 255).astype(np.uint8)
    if gt_vis.ndim == 2:
        gt_vis = np.stack([gt_vis, gt_vis, gt_vis], axis=-1)

    prob_vis = (pred_prob * 255).clip(0, 255).astype(np.uint8)
    if prob_vis.ndim == 2:
        # Colormap effect: blue to red
        heat = np.zeros((*prob_vis.shape, 3), dtype=np.uint8)
        heat[..., 0] = prob_vis  # Red channel
        prob_vis = heat

    pred_vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    pred_vis[pred_mask == 1] = [239, 68, 68]  # Bright red for change

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        axes[0].imshow(t1)
        axes[0].set_title(f"{scene_id} - T1 (Earlier)")
        axes[0].axis("off")

        axes[1].imshow(t2)
        axes[1].set_title(f"{scene_id} - T2 (Later)")
        axes[1].axis("off")

        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        axes[3].imshow(pred_prob, cmap="hot")
        axes[3].set_title("Prediction Probability")
        axes[3].axis("off")

        axes[4].imshow(pred_mask, cmap="Reds")
        axes[4].set_title("Predicted Binary Mask")
        axes[4].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved visual comparison artifact to: {output_path}")
    else:
        # Fallback: concatenate horizontal strip using PIL
        strip = np.concatenate([t1, t2, gt_vis, prob_vis, pred_vis], axis=1)
        Image.fromarray(strip).save(output_path)
        print(f"Saved concatenated visual comparison strip to: {output_path}")
