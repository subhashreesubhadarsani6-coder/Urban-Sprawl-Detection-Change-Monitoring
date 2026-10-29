"""Evaluation Metrics Module for Bi-temporal Satellite Change Detection.

Implements global pixel-level confusion matrix accumulation (TP, FP, TN, FN)
and computes Precision, Recall, F1-score, and Intersection-over-Union (IoU)
strictly over valid pixels.
Accumulates global counts across all batches/scenes before metric computation,
avoiding batch-averaging bias. Detects degenerate predictions (e.g. all zeros).
"""

from typing import Dict, Optional, Union
import torch
import numpy as np


class ChangeDetectionMetrics:
    """Accumulates confusion counts across batches or scenes and computes global metrics."""

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
        self.total_pixels = 0

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
            targets: Binary ground truth {0, 1} [B, 1, H, W] or [H, W]
            valid_mask: Optional boolean or {0, 1} mask of valid pixels
            threshold: Probability threshold for positive classification
        """
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
            self.valid_pixels += int(valid.sum())
            self.total_pixels += int(preds.size)
        else:
            pred_binary = pred_binary.flatten()
            targ_binary = targ_binary.flatten()
            self.valid_pixels += int(preds.size)
            self.total_pixels += int(preds.size)

        tp = int(np.sum((pred_binary == 1) & (targ_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (targ_binary == 0)))
        tn = int(np.sum((pred_binary == 0) & (targ_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (targ_binary == 1)))

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> Dict[str, float]:
        """Compute precision, recall, f1, iou, accuracy, and degeneracy diagnostics."""
        tp = float(self.tp)
        fp = float(self.fp)
        tn = float(self.tn)
        fn = float(self.fn)

        precision = tp / (tp + fp + self.eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + self.eps) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall + self.eps) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn + self.eps) if (tp + fp + fn) > 0 else 0.0
        total_eval = tp + fp + tn + fn
        accuracy = (tp + tn) / (total_eval + self.eps) if total_eval > 0 else 0.0

        # Diagnostics for degenerate models
        predicted_positives = tp + fp
        actual_positives = tp + fn
        is_all_zero = (predicted_positives == 0 and actual_positives > 0)
        is_all_one = (predicted_positives == total_eval and actual_positives < total_eval)

        return {
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1": float(round(f1, 4)),
            "iou": float(round(iou, 4)),
            "accuracy": float(round(accuracy, 4)),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "predicted_positives": int(predicted_positives),
            "actual_positives": int(actual_positives),
            "valid_pixels": int(self.valid_pixels),
            "is_all_zero": bool(is_all_zero),
            "is_all_one": bool(is_all_one),
            "degenerate": bool(is_all_zero or is_all_one)
        }
