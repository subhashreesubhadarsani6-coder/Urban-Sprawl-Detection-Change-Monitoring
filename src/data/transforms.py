"""Synchronized Spatial Transformations and Radiometric Normalization for Change Detection.

Guarantees that identical spatial operations (flips, rotations, crops) are applied
synchronously to T1, T2, ground-truth change mask, and valid-pixel mask.
"""

from typing import Tuple, Optional
import random
import numpy as np

class SynchronizedTransforms:
    """Synchronized random spatial augmentations for bi-temporal image pairs and masks."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

    def __call__(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        mask: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply identical random geometric augmentations across all inputs.

        Args:
            t1: [H, W, C]
            t2: [H, W, C]
            mask: [H, W] or [H, W, 1]
            valid_mask: Optional [H, W] or [H, W, 1]
        """
        # Horizontal flip
        if random.random() < self.hflip_prob:
            t1 = np.fliplr(t1)
            t2 = np.fliplr(t2)
            mask = np.fliplr(mask)
            if valid_mask is not None:
                valid_mask = np.fliplr(valid_mask)

        # Vertical flip
        if random.random() < self.vflip_prob:
            t1 = np.flipud(t1)
            t2 = np.flipud(t2)
            mask = np.flipud(mask)
            if valid_mask is not None:
                valid_mask = np.flipud(valid_mask)

        # 90-degree rotations (k in {1, 2, 3})
        if random.random() < self.rot90_prob:
            k = random.choice([1, 2, 3])
            t1 = np.rot90(t1, k, axes=(0, 1))
            t2 = np.rot90(t2, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
            if valid_mask is not None:
                valid_mask = np.rot90(valid_mask, k, axes=(0, 1))

        return (
            np.ascontiguousarray(t1),
            np.ascontiguousarray(t2),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(valid_mask) if valid_mask is not None else None
        )


def normalize_imagery(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize imagery array from [0, 255] or [0, 1] to standard distribution.
    Expects input shape [H, W, C] and returns normalized float32 [H, W, C].
    """
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    normalized = (img - mean_arr) / (std_arr + 1e-7)
    return normalized


def denormalize_imagery(
    img: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Invert normalization back to [0, 255] uint8 RGB."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    denorm = (img * std_arr) + mean_arr
    denorm = np.clip(denorm * 255.0, 0, 255).astype(np.uint8)
    return denorm
