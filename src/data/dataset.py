"""Change Detection Dataset Module.

Provides ChangeDetectionDataset supporting both Parquet-based OSCD data and
extracted scene directories. Supports patch extraction, synchronized augmentation,
normalization, valid-pixel masking, and geographic scene filtering.
"""

from typing import Dict, List, Optional, Tuple, Union
import io
import os
from PIL import Image
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .transforms import SynchronizedTransforms, normalize_imagery

# Scene order mapping for the 14 train and 10 test scenes in single-shard OSCD parquet files
OSCD_TRAIN_SCENE_ORDER = [
    "nantes", "pisa", "rennes", "cupertino", "aguasclaras",
    "bordeaux", "beihai", "hongkong", "beirut", "paris",
    "mumbai", "saclay_e", "bercy", "abudhabi"
]

OSCD_TEST_SCENE_ORDER = [
    "rio", "saclay_w", "valencia", "chongqing", "montpellier",
    "brasilia", "norcia", "dubai", "milano", "lasvegas"
]

class ChangeDetectionDataset(Dataset):
    """PyTorch Dataset for Bi-temporal Satellite Change Detection.

    Supports:
    - Bounded-memory reading from single-shard or multi-shard Parquet files.
    - Synchronized spatial data augmentations.
    - Systematic patch extraction with configurable stride and size.
    - Full-scene evaluation mode.
    - Strict geographic scene filtering (only loads scenes belonging to the specified split).
    """

    def __init__(
        self,
        parquet_path: str,
        split_scenes: Optional[List[str]] = None,
        is_train_split: bool = True,
        patch_size: Optional[int] = 256,
        stride: Optional[int] = 128,
        min_valid_ratio: float = 0.5,
        augment: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Args:
            parquet_path: Path to the parquet file (e.g., train-00000-of-00001.parquet).
            split_scenes: List of scene names allowed in this dataset instance.
            is_train_split: True if loading training scenes, False for test scenes.
            patch_size: Square patch size for training (None for full-scene mode).
            stride: Stride for patch sliding window. Defaults to patch_size // 2.
            min_valid_ratio: Minimum ratio of valid pixels required to include a patch.
            augment: Whether to apply synchronized spatial augmentations.
            mean: Normalization channel means.
            std: Normalization channel standard deviations.
        """
        self.parquet_path = parquet_path
        self.split_scenes = set(split_scenes) if split_scenes else None
        self.is_train_split = is_train_split
        self.patch_size = patch_size
        self.stride = stride or (patch_size // 2 if patch_size else None)
        self.min_valid_ratio = min_valid_ratio
        self.augment = augment
        self.mean = mean
        self.std = std

        self.transform = SynchronizedTransforms() if augment else None

        # Determine scene name mapping based on split
        self.scene_names = OSCD_TRAIN_SCENE_ORDER if is_train_split else OSCD_TEST_SCENE_ORDER

        # Load table
        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()

        # Cache decoded full scenes for scenes matching this split
        self.scenes: Dict[str, Dict[str, np.ndarray]] = {}
        self.patches: List[Tuple[str, int, int]] = []  # (scene_id, top, left)

        self._load_scenes_and_index_patches()

    def _load_scenes_and_index_patches(self):
        """Decode imagery and build patch index for all authorized scenes."""
        for row_idx in range(len(self.df)):
            scene_name = self.scene_names[row_idx] if row_idx < len(self.scene_names) else f"scene_{row_idx}"

            # If geographic split filter is active, skip unauthorized scenes
            if self.split_scenes is not None and scene_name not in self.split_scenes:
                continue

            # Decode T1, T2, mask
            row = self.df.iloc[row_idx]
            img1_bytes = row["image1"]["bytes"]
            img2_bytes = row["image2"]["bytes"]
            mask_bytes = row["mask"]["bytes"]

            img1 = np.array(Image.open(io.BytesIO(img1_bytes)).convert("RGB"))
            img2 = np.array(Image.open(io.BytesIO(img2_bytes)).convert("RGB"))
            raw_mask = np.array(Image.open(io.BytesIO(mask_bytes)))

            # Convert mask to binary {0, 1}
            binary_mask = (raw_mask > 0).astype(np.uint8)

            # Valid pixel mask (exclude nodata: e.g. purely black boundary pixels in all channels)
            # Sentinel-2 nodata often has 0 across all RGB channels in both dates
            nodata_t1 = (img1.sum(axis=-1) == 0)
            nodata_t2 = (img2.sum(axis=-1) == 0)
            valid_mask = (~(nodata_t1 & nodata_t2)).astype(np.uint8)

            h, w = binary_mask.shape

            self.scenes[scene_name] = {
                "t1": img1,
                "t2": img2,
                "mask": binary_mask,
                "valid_mask": valid_mask,
                "height": h,
                "width": w
            }

            # Generate patches if patch_size specified
            if self.patch_size is not None:
                ps = self.patch_size
                st = self.stride
                for top in range(0, max(1, h - ps + 1), st):
                    for left in range(0, max(1, w - ps + 1), st):
                        # Ensure patch fits inside scene bounds
                        actual_top = min(top, max(0, h - ps))
                        actual_left = min(left, max(0, w - ps))
                        patch_valid = valid_mask[actual_top:actual_top + ps, actual_left:actual_left + ps]
                        if patch_valid.mean() >= self.min_valid_ratio:
                            self.patches.append((scene_name, actual_top, actual_left))
            else:
                # Full scene mode
                self.patches.append((scene_name, 0, 0))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        scene_name, top, left = self.patches[idx]
        scene = self.scenes[scene_name]

        if self.patch_size is not None:
            ps = self.patch_size
            t1_patch = scene["t1"][top:top + ps, left:left + ps]
            t2_patch = scene["t2"][top:top + ps, left:left + ps]
            mask_patch = scene["mask"][top:top + ps, left:left + ps]
            valid_patch = scene["valid_mask"][top:top + ps, left:left + ps]
        else:
            t1_patch = scene["t1"]
            t2_patch = scene["t2"]
            mask_patch = scene["mask"]
            valid_patch = scene["valid_mask"]

        # Apply synchronized spatial augmentations
        if self.transform is not None:
            t1_patch, t2_patch, mask_patch, valid_patch = self.transform(
                t1_patch, t2_patch, mask_patch, valid_patch
            )

        # Normalize imagery
        t1_norm = normalize_imagery(t1_patch, self.mean, self.std)
        t2_norm = normalize_imagery(t2_patch, self.mean, self.std)

        # Convert to PyTorch tensors [C, H, W]
        t1_tensor = torch.from_numpy(t1_norm.transpose(2, 0, 1)).float()
        t2_tensor = torch.from_numpy(t2_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).float()
        valid_tensor = torch.from_numpy(valid_patch).unsqueeze(0).float()

        return {
            "image_t1": t1_tensor,
            "image_t2": t2_tensor,
            "mask": mask_tensor,
            "valid_mask": valid_tensor,
            "scene_id": scene_name
        }
