"""Unit tests for Change Detection Metrics Module."""

import pytest
import numpy as np
import torch
from src.metrics import ChangeDetectionMetrics

def test_exact_confusion_counts():
    # Construct exact known 4-pixel case:
    # Pred:   [1, 1, 0, 0]
    # Target: [1, 0, 1, 0]
    # TP = 1 (idx 0)
    # FP = 1 (idx 1)
    # FN = 1 (idx 2)
    # TN = 1 (idx 3)
    preds = np.array([1.0, 1.0, 0.0, 0.0])
    targets = np.array([1.0, 0.0, 1.0, 0.0])

    tracker = ChangeDetectionMetrics()
    tracker.update(preds, targets, threshold=0.5)
    m = tracker.compute()

    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["valid_pixels"] == 4

    # Precision: 1 / (1 + 1) = 0.5
    assert m["precision"] == 0.5
    # Recall: 1 / (1 + 1) = 0.5
    assert m["recall"] == 0.5
    # F1: 2 * 1 / (2 + 1 + 1) = 0.5
    assert m["f1"] == 0.5
    # IoU: 1 / (1 + 1 + 1) = 1/3 ~ 0.33333
    assert abs(m["iou"] - 1/3) < 1e-4
    # Accuracy: (1 + 1) / 4 = 0.5
    assert m["accuracy"] == 0.5

def test_empty_change_metrics():
    # Both pred and targets are all negative (0)
    preds = np.zeros(100)
    targets = np.zeros(100)

    tracker = ChangeDetectionMetrics()
    tracker.update(preds, targets, threshold=0.5)
    m = tracker.compute()

    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 100
    assert m["f1"] == 1.0  # Safe handling of perfect true negative agreement
    assert m["iou"] == 1.0

def test_valid_mask_in_metrics():
    # 4 pixels, but only first 2 are valid
    preds = np.array([1.0, 0.0, 1.0, 1.0])
    targets = np.array([1.0, 0.0, 0.0, 0.0])
    valid_mask = np.array([1, 1, 0, 0])

    tracker = ChangeDetectionMetrics()
    tracker.update(preds, targets, valid_mask=valid_mask, threshold=0.5)
    m = tracker.compute()

    assert m["valid_pixels"] == 2
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0

def test_batch_aggregation():
    # Verify that updating 2 separate batches gives identical global counts as 1 merged batch
    tracker = ChangeDetectionMetrics()
    b1_p = np.array([1.0, 0.0])
    b1_t = np.array([1.0, 0.0])
    b2_p = np.array([1.0, 0.0])
    b2_t = np.array([0.0, 1.0])

    tracker.update(b1_p, b1_t)
    tracker.update(b2_p, b2_t)
    m = tracker.compute()

    assert m["valid_pixels"] == 4
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
