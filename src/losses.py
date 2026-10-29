"""Combined Loss Functions for Binary Change Detection with Class Imbalance & Masking.

Implements BCEWithLogitsLoss combined with Soft Dice Loss, fully aware of valid-pixel masks,
imbalance pos_weighting, and numerical stability for empty masks.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """Numerically stable Soft Dice Loss with valid-pixel masking support."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Soft Dice loss.

        Args:
            logits: Predicted raw logits [B, 1, H, W]
            targets: Binary ground truth {0.0, 1.0} [B, 1, H, W]
            valid_mask: Optional boolean or float mask of valid pixels [B, 1, H, W]

        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)

        if valid_mask is not None:
            probs = probs * valid_mask
            targets = targets * valid_mask

        # Flatten per batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat + targets_flat, dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        loss = 1.0 - dice_score
        return loss.mean()


class CombinedBceDiceLoss(nn.Module):
    """Combined BCEWithLogitsLoss + SoftDiceLoss for imbalanced change segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice_loss = SoftDiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted combined loss."""
        # 1. Masked BCE Loss
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)

        if valid_mask is not None:
            # If all pixels are invalid, return safe zero loss
            valid_count = torch.sum(valid_mask)
            if valid_count == 0:
                return (logits * 0.0).sum()

            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight_tensor, reduction='none'
            )
            bce = (bce * valid_mask).sum() / (valid_count + 1e-7)
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight_tensor, reduction='mean'
            )

        # 2. Soft Dice Loss
        dice = self.dice_loss(logits, targets, valid_mask=valid_mask)

        # 3. Weighted total loss
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        return total_loss
