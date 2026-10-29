"""Unit tests for Change Detection Dataset & Geographic Splits."""

import os
import pytest
import numpy as np
import torch
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits
from src.data.transforms import SynchronizedTransforms

def test_geographic_scene_isolation():
    splits = create_geographic_splits(
        seed=42,
        output_path="configs/test_splits.json"
    )
    train_s = set(splits["train"])
    val_s = set(splits["val"])
    test_s = set(splits["test"])

    assert len(train_s) > 0
    assert len(val_s) > 0
    assert len(test_s) > 0

    # Ensure zero overlap
    assert not (train_s & val_s), "Train and Val share scenes!"
    assert not (train_s & test_s), "Train and Test share scenes!"
    assert not (val_s & test_s), "Val and Test share scenes!"

    if os.path.exists("configs/test_splits.json"):
        os.remove("configs/test_splits.json")

def test_synchronized_transforms():
    t = SynchronizedTransforms(hflip_prob=1.0, vflip_prob=1.0, rot90_prob=0.0)
    img1 = np.arange(16).reshape(4, 4, 1)
    img2 = np.arange(16).reshape(4, 4, 1) + 10
    mask = np.arange(16).reshape(4, 4)

    t1_out, t2_out, mask_out, _ = t(img1, img2, mask)

    # Flipping left-right then up-down is equivalent to 180 rotation
    expected_mask = np.flipud(np.fliplr(mask))
    assert np.array_equal(mask_out, expected_mask)
    assert np.array_equal(t1_out[..., 0], np.flipud(np.fliplr(img1[..., 0])))
    assert np.array_equal(t2_out[..., 0], np.flipud(np.fliplr(img2[..., 0])))

def test_dataset_item_structure():
    parquet_path = "dataset/train-00000-of-00001.parquet"
    if not os.path.exists(parquet_path):
        pytest.skip("Dataset file not available")

    # Load 1 scene for fast testing
    ds = ChangeDetectionDataset(
        parquet_path=parquet_path,
        split_scenes=["bercy"],  # Smallest scene (360x395)
        is_train_split=True,
        patch_size=128,
        stride=128
    )

    assert len(ds) > 0
    sample = ds[0]

    assert "image_t1" in sample
    assert "image_t2" in sample
    assert "mask" in sample
    assert "valid_mask" in sample
    assert "scene_id" in sample

    assert sample["image_t1"].shape == (3, 128, 128)
    assert sample["image_t2"].shape == (3, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)
    assert sample["valid_mask"].shape == (1, 128, 128)
    assert sample["scene_id"] == "bercy"
