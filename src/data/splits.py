"""Geographic Scene Splitting Module for Change Detection.

Enforces scene-level geographic isolation: all patches from a city/scene remain
strictly inside that split to prevent geographic data leakage.
"""

from typing import Dict, List, Optional
import os
import json
import random

OFFICIAL_TRAIN_SCENES = [
    "nantes", "pisa", "rennes", "cupertino", "aguasclaras",
    "bordeaux", "beihai", "hongkong", "beirut", "paris",
    "mumbai", "saclay_e", "bercy", "abudhabi"
]

OFFICIAL_TEST_SCENES = [
    "rio", "saclay_w", "valencia", "chongqing", "montpellier",
    "brasilia", "norcia", "dubai", "milano", "lasvegas"
]

def create_geographic_splits(
    train_scenes: Optional[List[str]] = None,
    test_scenes: Optional[List[str]] = None,
    val_fraction: float = 0.22,  # 3 of 14 train scenes for validation (~21.4%)
    seed: int = 42,
    output_path: str = "configs/geographic_splits.json"
) -> Dict[str, List[str]]:
    """Generate or retrieve deterministic geographic scene splits.

    Args:
        train_scenes: List of scene names available for train/val partitioning.
        test_scenes: List of strictly held-out test scenes.
        val_fraction: Fraction of train_scenes to allocate to validation.
        seed: Random seed for reproducible scene selection.
        output_path: File path to persist and reload the split manifest.

    Returns:
        Dictionary with keys 'train', 'val', and 'test'.
    """
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            splits = json.load(f)
            # Verify no geographic leakage
            train_set = set(splits.get("train", []))
            val_set = set(splits.get("val", []))
            test_set = set(splits.get("test", []))
            assert not (train_set & val_set), f"Geographic leakage between train and val: {train_set & val_set}"
            assert not (train_set & test_set), f"Geographic leakage between train and test: {train_set & test_set}"
            assert not (val_set & test_set), f"Geographic leakage between val and test: {val_set & test_set}"
            return splits

    if train_scenes is None:
        train_scenes = list(OFFICIAL_TRAIN_SCENES)
    if test_scenes is None:
        test_scenes = list(OFFICIAL_TEST_SCENES)

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_train = list(train_scenes)
    rng.shuffle(shuffled_train)

    num_val = max(1, int(round(len(train_scenes) * val_fraction)))
    val_selected = sorted(shuffled_train[:num_val])
    train_selected = sorted(shuffled_train[num_val:])
    test_selected = sorted(test_scenes)

    splits = {
        "metadata": {
            "description": "Geographically isolated scene splits for OSCD change detection",
            "seed": seed,
            "total_scenes": len(train_selected) + len(val_selected) + len(test_selected),
            "train_scenes_count": len(train_selected),
            "val_scenes_count": len(val_selected),
            "test_scenes_count": len(test_selected)
        },
        "train": train_selected,
        "val": val_selected,
        "test": test_selected
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits
