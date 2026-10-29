"""Inference Predictor for Bi-temporal Satellite Change Detection.

Provides full-scene and tiled inference using a verified trained checkpoint.
Supports seamless overlap blending, thresholding, and nodata masking.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

from ..models.siamese_unet import FCSiamDiff
from ..data.transforms import normalize_imagery
from ..training.checkpoint import load_checkpoint

class ChangePredictor:
    """Production predictor for FCSiamDiff change detection checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.checkpoint_path = checkpoint_path

        # Initial dummy model to read parameters from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        in_channels = checkpoint.get("in_channels", 3)
        base_channels = checkpoint.get("base_channels", 32)
        num_classes = checkpoint.get("num_classes", 1)

        self.model = FCSiamDiff(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes
        ).to(self.device)

        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        self.threshold = threshold if threshold is not None else checkpoint.get("threshold", 0.5)
        self.metadata = {
            "checkpoint": checkpoint_path,
            "in_channels": in_channels,
            "base_channels": base_channels,
            "threshold": self.threshold,
            "best_metric": checkpoint.get("best_metric"),
            "epoch": checkpoint.get("epoch")
        }

    def predict_patch(
        self,
        t1_patch: np.ndarray,
        t2_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict change on a single spatial patch [H, W, 3].

        Returns:
            (binary_mask [H, W], probability_map [H, W])
        """
        t1_norm = normalize_imagery(t1_patch)
        t2_norm = normalize_imagery(t2_patch)

        t1_tensor = torch.from_numpy(t1_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        t2_tensor = torch.from_numpy(t2_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(t1_tensor, t2_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary = (probs >= self.threshold).astype(np.uint8)
        return binary, probs

    def predict_scene(
        self,
        t1_scene: np.ndarray,
        t2_scene: np.ndarray,
        patch_size: int = 256,
        overlap: int = 64
    ) -> Dict[str, Any]:
        """Run seamless tiled inference over a large scene using sliding window blending."""
        h, w, c = t1_scene.shape
        stride = patch_size - overlap

        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # 2D Hann window for smooth seam blending
        hann_1d = np.hanning(patch_size)
        hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
        hann_2d = np.maximum(hann_2d, 1e-4)

        for top in range(0, h, stride):
            for left in range(0, w, stride):
                # Clamp patch window to scene boundaries
                actual_top = min(top, max(0, h - patch_size))
                actual_left = min(left, max(0, w - patch_size))
                actual_h = min(patch_size, h - actual_top)
                actual_w = min(patch_size, w - actual_left)

                t1_p = t1_scene[actual_top:actual_top + actual_h, actual_left:actual_left + actual_w]
                t2_p = t2_scene[actual_top:actual_top + actual_h, actual_left:actual_left + actual_w]

                # Pad to square patch_size if on edge
                pad_h = patch_size - actual_h
                pad_w = patch_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    t1_p = np.pad(t1_p, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                    t2_p = np.pad(t2_p, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                _, p_prob = self.predict_patch(t1_p, t2_p)
                p_prob = p_prob[:actual_h, :actual_w]
                p_weight = hann_2d[:actual_h, :actual_w]

                prob_map[actual_top:actual_top + actual_h, actual_left:actual_left + actual_w] += (p_prob * p_weight)
                weight_map[actual_top:actual_top + actual_h, actual_left:actual_left + actual_w] += p_weight

        # Normalize probabilities by accumulated window weights
        prob_map = prob_map / np.maximum(weight_map, 1e-6)
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        # Mask out nodata pixels
        nodata = (t1_scene.sum(axis=-1) == 0) & (t2_scene.sum(axis=-1) == 0)
        binary_mask[nodata] = 0
        prob_map[nodata] = 0.0

        return {
            "binary_mask": binary_mask,
            "probability_map": prob_map,
            "threshold": self.threshold,
            "changed_pixels": int(np.sum(binary_mask)),
            "total_pixels": int(h * w),
            "change_percentage": round(float(np.mean(binary_mask)) * 100, 2),
            "metadata": self.metadata
        }
