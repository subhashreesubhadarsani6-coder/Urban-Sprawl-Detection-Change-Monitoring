"""Unit tests for Change Detection Loss Functions."""

import pytest
import torch
from src.losses import SoftDiceLoss, CombinedBceDiceLoss

def test_finite_loss_and_backprop():
    criterion = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

def test_empty_mask_handling():
    # Targets contain zero changes (all 0.0)
    criterion = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.zeros((2, 1, 64, 64)).float()

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

def test_valid_mask_exclusion():
    # Only the first quadrant is valid
    criterion = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    targets = torch.ones(1, 1, 32, 32).float()

    valid_mask = torch.zeros(1, 1, 32, 32).float()
    valid_mask[:, :, :16, :16] = 1.0  # Top-left quadrant valid

    loss = criterion(logits, targets, valid_mask=valid_mask)
    assert torch.isfinite(loss)
    loss.backward()

    # Pixels outside the valid quadrant should have zero gradients
    outside_grad = logits.grad[:, :, 16:, 16:]
    assert torch.all(outside_grad == 0.0), "Gradients leaked into invalid pixel regions!"

def test_all_invalid_pixels_safe():
    criterion = CombinedBceDiceLoss()
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    targets = torch.ones(1, 1, 32, 32).float()
    valid_mask = torch.zeros(1, 1, 32, 32).float()

    loss = criterion(logits, targets, valid_mask=valid_mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
