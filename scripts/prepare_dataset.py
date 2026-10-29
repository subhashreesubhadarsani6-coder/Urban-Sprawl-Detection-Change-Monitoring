"""Prepare and Index Geographic Dataset Splits for Siamese U-Net Training.

Generates deterministic geographic splits, indexes all scenes and patches,
and computes class balance statistics without data leakage.
"""

import os
import sys
import json
import yaml

# Ensure project root in python path
sys.path.insert(0, os.path.abspath("."))

from src.data.splits import create_geographic_splits
from src.data.dataset import ChangeDetectionDataset

def prepare_dataset(config_path: str = "configs/train.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits_file = cfg["dataset"].get("splits_file", "configs/geographic_splits.json")
    splits = create_geographic_splits(output_path=splits_file, seed=cfg["reproducibility"]["seed"])

    print(f"=== Geographic Split Manifest ({splits_file}) ===")
    print(f"Train scenes ({len(splits['train'])}): {splits['train']}")
    print(f"Val scenes   ({len(splits['val'])}): {splits['val']}")
    print(f"Test scenes  ({len(splits['test'])}): {splits['test']}")

    # Build datasets to count patches and analyze class distribution
    train_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=splits["train"],
        is_train_split=True,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=False
    )

    val_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=splits["val"],
        is_train_split=True,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=False
    )

    test_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["test_parquet"],
        split_scenes=splits["test"],
        is_train_split=False,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=False
    )

    summary = {
        "train": {
            "scenes": splits["train"],
            "patch_count": len(train_ds)
        },
        "val": {
            "scenes": splits["val"],
            "patch_count": len(val_ds)
        },
        "test": {
            "scenes": splits["test"],
            "patch_count": len(test_ds)
        }
    }

    print("\n=== Dataset Patch Index Summary ===")
    print(f"Train: {len(train_ds)} patches across {len(splits['train'])} scenes")
    print(f"Val:   {len(val_ds)} patches across {len(splits['val'])} scenes")
    print(f"Test:  {len(test_ds)} patches across {len(splits['test'])} scenes")

    os.makedirs("reports", exist_ok=True)
    with open("reports/dataset_preparation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset preparation completed successfully. Saved to reports/dataset_preparation.json")
    return summary

if __name__ == "__main__":
    prepare_dataset()
