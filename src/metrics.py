"""Evaluation Metrics Module for Change Detection.

Implements pixel-level confusion accumulation (TP, FP, TN, FN) and computes
Precision, Recall, F1-Score, IoU, and Accuracy strictly over valid pixels.
Aggregates global counts before metric calculation to avoid batch-averaging bias.
"""

from typing import Dict, Optional, Union
import torch
import numpy as np

class ChangeDetectionMetrics:
    """Accumulates confusion counts across batches/scenes and computes global metrics."""

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset running confusion counts."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.valid_pixels = 0

    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        threshold: float = 0.5
    ):
        """Update confusion counts with a batch or scene.

        Args:
            predictions: Predicted logits or probabilities [B, 1, H, W] or [H, W]
            targets: Binary ground truth {0, 1}
            valid_mask: Optional boolean or {0, 1} mask of valid pixels
            threshold: Probability threshold for positive classification
        """
        # Convert tensors to numpy if needed
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy()
        else:
            preds = np.asarray(predictions)

        if isinstance(targets, torch.Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.asarray(targets)

        # Convert logits to probabilities if outside [0, 1]
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = 1.0 / (1.0 + np.exp(-preds))

        pred_binary = (preds >= threshold).astype(np.int64)
        targ_binary = (targs >= 0.5).astype(np.int64)

        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                v_mask = valid_mask.detach().cpu().numpy()
            else:
                v_mask = np.asarray(valid_mask)
            valid = (v_mask > 0.5).astype(bool)
            pred_binary = pred_binary[valid]
            targ_binary = targ_binary[valid]
        else:
            pred_binary = pred_binary.flatten()
            targ_binary = targ_binary.flatten()

        tp = int(np.sum((pred_binary == 1) & (targ_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (targ_binary == 0)))
        tn = int(np.sum((pred_binary == 0) & (targ_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (targ_binary == 1)))

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.valid_pixels += (tp + fp + tn + fn)

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics over all accumulated samples."""
        denom_prec = self.tp + self.fp
        precision = self.tp / denom_prec if denom_prec > 0 else (1.0 if self.fn == 0 else 0.0)

        denom_rec = self.tp + self.fn
        recall = self.tp / denom_rec if denom_rec > 0 else (1.0 if self.fp == 0 else 0.0)

        denom_f1 = (2 * self.tp) + self.fp + self.fn
        f1 = (2.0 * self.tp) / denom_f1 if denom_f1 > 0 else (1.0 if (self.fp == 0 and self.fn == 0) else 0.0)

        denom_iou = self.tp + self.fp + self.fn
        iou = self.tp / denom_iou if denom_iou > 0 else (1.0 if (self.fp == 0 and self.fn == 0) else 0.0)

        denom_acc = self.valid_pixels
        accuracy = (self.tp + self.tn) / denom_acc if denom_acc > 0 else 0.0
        prevalence = denom_rec / denom_acc if denom_acc > 0 else 0.0

        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "valid_pixels": self.valid_pixels,
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "f1": round(float(f1), 5),
            "iou": round(float(iou), 5),
            "accuracy": round(float(accuracy), 5),
            "prevalence": round(float(prevalence), 5)
        }


def compute_metrics_at_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: list,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[float, Dict[str, float]]:
    """Evaluate metric values across multiple candidate thresholds."""
    results = {}
    for thresh in thresholds:
        tracker = ChangeDetectionMetrics()
        tracker.update(probs, targets, valid_mask=valid_mask, threshold=thresh)
        results[round(thresh, 3)] = tracker.compute()
    return results
