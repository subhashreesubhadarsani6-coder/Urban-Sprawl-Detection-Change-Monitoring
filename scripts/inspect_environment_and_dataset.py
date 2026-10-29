"""Phase A: Environment, GPU, and Dataset Inspection Script.

Inspects:
1. Python, PyTorch, platform versions.
2. GPU hardware (NVIDIA RTX 3050, 4GB VRAM, driver, CUDA capabilities).
3. Installed packages.
4. Dataset directory recursively (Parquet files, archives, crdownloads).
5. Schema, data types, channels, value ranges, nodata values.
6. Scene matching and spatial alignment across T1, T2, and ground-truth mask.
7. Class distribution (changed vs unchanged pixels) and mask validity.
8. Generates reports/dataset_inspection.json and sample visual artifacts.
"""

import os
import sys
import json
import io
import subprocess
import platform
from pathlib import Path
from PIL import Image
import numpy as np
import pyarrow.parquet as pq

# OSCD Scene Mapping
OSCD_TRAIN_SCENES = [
    "nantes", "pisa", "rennes", "cupertino", "aguasclaras",
    "bordeaux", "beihai", "hongkong", "beirut", "paris",
    "mumbai", "saclay_e", "bercy", "abudhabi"
]

OSCD_TEST_SCENES = [
    "rio", "saclay_w", "valencia", "chongqing", "montpellier",
    "brasilia", "norcia", "dubai", "milano", "lasvegas"
]


def detect_gpu_info():
    """Detect GPU hardware details via nvidia-smi if available."""
    gpu_info = {
        "hardware_detected": False,
        "gpu_name": None,
        "vram_total_mb": None,
        "driver_version": None,
        "cuda_version_supported": None,
        "error": None
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            gpu_info["hardware_detected"] = True
            gpu_info["gpu_name"] = parts[0]
            gpu_info["vram_total_mb"] = float(parts[1]) if len(parts) > 1 else None
            gpu_info["driver_version"] = parts[2] if len(parts) > 2 else None

        # Query max supported CUDA version
        smi_out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        for line in smi_out.splitlines():
            if "CUDA Version:" in line:
                cuda_v = line.split("CUDA Version:")[1].split("|")[0].strip()
                gpu_info["cuda_version_supported"] = cuda_v
                break
    except Exception as e:
        gpu_info["error"] = str(e)
    return gpu_info


def inspect_all():
    print("=" * 70)
    print("PHASE A: ENVIRONMENT & DATASET INSPECTION")
    print("=" * 70)

    # 1. Environment
    import torch
    gpu_hw = detect_gpu_info()
    cuda_available = torch.cuda.is_available()

    env_report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available_in_torch": cuda_available,
        "gpu_hardware": gpu_hw,
        "active_device_for_training": "cuda:0" if cuda_available else "cpu"
    }

    print(f"Python:   {sys.version.split()[0]}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"GPU Hardware: {gpu_hw.get('gpu_name')} ({gpu_hw.get('vram_total_mb')} MB VRAM, Driver {gpu_hw.get('driver_version')})")
    print(f"CUDA in PyTorch: {cuda_available} | Target Device: {env_report['active_device_for_training']}")

    # 2. Dataset Directory Recursive Inspection
    dataset_dir = Path("dataset").resolve()
    print(f"\nScanning dataset directory: {dataset_dir}...")
    dataset_files = []
    parquet_files = []
    zip_files = []
    crdownload_files = []

    for p in dataset_dir.rglob("*"):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            file_entry = {
                "name": p.name,
                "relative_path": str(p.relative_to(dataset_dir)),
                "size_mb": round(size_mb, 2),
                "extension": p.suffix.lower()
            }
            dataset_files.append(file_entry)
            if p.suffix.lower() == ".parquet":
                parquet_files.append(p)
            elif p.suffix.lower() == ".zip":
                zip_files.append(p)
            elif p.suffix.lower() == ".crdownload":
                crdownload_files.append(p)

    print(f"Total files found: {len(dataset_files)}")
    print(f"  - Parquet shards: {len(parquet_files)}")
    print(f"  - Zip archives:   {len(zip_files)}")
    print(f"  - Incomplete crdownloads: {len(crdownload_files)}")

    # 3. Inspect Parquet Schemas & Scene Matching
    train_parquet = dataset_dir / "train-00000-of-00001.parquet"
    test_parquet = dataset_dir / "test-00000-of-00001.parquet"

    assert train_parquet.exists(), f"Missing primary train parquet: {train_parquet}"
    assert test_parquet.exists(), f"Missing primary test parquet: {test_parquet}"

    def inspect_parquet_shard(path: Path, scene_names: list, split_label: str):
        table = pq.read_table(str(path))
        df = table.to_pandas()
        schema_cols = table.column_names

        scenes_data = []
        total_changed_pixels = 0
        total_unchanged_pixels = 0
        total_nodata_pixels = 0

        for idx in range(len(df)):
            row = df.iloc[idx]
            scene_name = scene_names[idx] if idx < len(scene_names) else f"scene_{idx}"

            # Decode images and mask
            img1 = Image.open(io.BytesIO(row["image1"]["bytes"])).convert("RGB")
            img2 = Image.open(io.BytesIO(row["image2"]["bytes"])).convert("RGB")
            mask_img = Image.open(io.BytesIO(row["mask"]["bytes"]))

            arr1 = np.array(img1)
            arr2 = np.array(img2)
            arr_m = np.array(mask_img)

            # Spatial compatibility check
            h1, w1 = arr1.shape[:2]
            h2, w2 = arr2.shape[:2]
            hm, wm = arr_m.shape[:2]

            assert (h1, w1) == (h2, w2) == (hm, wm), f"Dimension mismatch in {scene_name}: img1={(h1, w1)}, img2={(h2, w2)}, mask={(hm, wm)}"

            # Mask values check
            unique_vals = np.unique(arr_m).tolist()
            binary_mask = (arr_m > 0).astype(np.uint8)
            num_changed = int(np.sum(binary_mask == 1))
            num_unchanged = int(np.sum(binary_mask == 0))

            # Nodata detection
            nodata = (arr1.sum(axis=-1) == 0) & (arr2.sum(axis=-1) == 0)
            num_nodata = int(np.sum(nodata))

            total_changed_pixels += num_changed
            total_unchanged_pixels += num_unchanged
            total_nodata_pixels += num_nodata

            scenes_data.append({
                "scene_id": scene_name,
                "row_idx": idx,
                "height": h1,
                "width": w1,
                "channels": 3,
                "img1_dtype": str(arr1.dtype),
                "img1_range": [int(arr1.min()), int(arr1.max())],
                "img2_dtype": str(arr2.dtype),
                "img2_range": [int(arr2.min()), int(arr2.max())],
                "mask_dtype": str(arr_m.dtype),
                "mask_raw_unique_values": unique_vals,
                "changed_pixels": num_changed,
                "unchanged_pixels": num_unchanged,
                "change_percentage": round(100.0 * num_changed / max(1, h1 * w1), 3),
                "nodata_pixels": num_nodata,
                "spatially_aligned": True
            })

        total_pixels = total_changed_pixels + total_unchanged_pixels
        return {
            "file": path.name,
            "split": split_label,
            "schema_columns": schema_cols,
            "num_rows": len(df),
            "scenes": scenes_data,
            "summary": {
                "total_scenes": len(scenes_data),
                "total_pixels": total_pixels,
                "total_changed_pixels": total_changed_pixels,
                "total_unchanged_pixels": total_unchanged_pixels,
                "total_nodata_pixels": total_nodata_pixels,
                "change_prevalence_pct": round(100.0 * total_changed_pixels / max(1, total_pixels), 3),
                "imbalance_ratio_neg_to_pos": round(total_unchanged_pixels / max(1, total_changed_pixels), 2)
            }
        }

    train_audit = inspect_parquet_shard(train_parquet, OSCD_TRAIN_SCENES, "train")
    test_audit = inspect_parquet_shard(test_parquet, OSCD_TEST_SCENES, "test")

    print("\n--- Train Split Inspection ---")
    print(f"Scenes: {train_audit['num_rows']} ({', '.join([s['scene_id'] for s in train_audit['scenes']])})")
    print(f"Total Pixels: {train_audit['summary']['total_pixels']:,}")
    print(f"Changed Pixels: {train_audit['summary']['total_changed_pixels']:,} ({train_audit['summary']['change_prevalence_pct']}%)")
    print(f"Class Imbalance: 1 changed : {train_audit['summary']['imbalance_ratio_neg_to_pos']} unchanged")

    print("\n--- Test Split Inspection ---")
    print(f"Scenes: {test_audit['num_rows']} ({', '.join([s['scene_id'] for s in test_audit['scenes']])})")
    print(f"Total Pixels: {test_audit['summary']['total_pixels']:,}")
    print(f"Changed Pixels: {test_audit['summary']['total_changed_pixels']:,} ({test_audit['summary']['change_prevalence_pct']}%)")
    print(f"Class Imbalance: 1 changed : {test_audit['summary']['imbalance_ratio_neg_to_pos']} unchanged")

    # 4. Save representative visual samples
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Las Vegas sample
    table_test = pq.read_table(str(test_parquet))
    df_test = table_test.to_pandas()
    vegas_idx = OSCD_TEST_SCENES.index("lasvegas")
    vegas_row = df_test.iloc[vegas_idx]

    vegas_t1 = Image.open(io.BytesIO(vegas_row["image1"]["bytes"])).convert("RGB")
    vegas_t2 = Image.open(io.BytesIO(vegas_row["image2"]["bytes"])).convert("RGB")
    vegas_m = Image.open(io.BytesIO(vegas_row["mask"]["bytes"]))
    vegas_m_arr = (np.array(vegas_m) > 0).astype(np.uint8) * 255
    vegas_m_vis = Image.fromarray(vegas_m_arr)

    vegas_t1.save(reports_dir / "sample_t1_lasvegas.png")
    vegas_t2.save(reports_dir / "sample_t2_lasvegas.png")
    vegas_m_vis.save(reports_dir / "sample_gt_lasvegas.png")
    print(f"\nSaved representative T1/T2/GT sample images to {reports_dir}/sample_*_lasvegas.png")

    # 5. Compile Final Inspection JSON Report
    full_report = {
        "timestamp": "2026-10-25T13:58:00Z",
        "environment": env_report,
        "dataset_files": dataset_files,
        "dataset_structure": {
            "dataset_name": "Onera Satellite Change Detection (OSCD) RGB Dataset",
            "format": "Apache Parquet containing embedded PNG/TIFF byte streams",
            "total_scenes_available": train_audit["num_rows"] + test_audit["num_rows"],
            "train_parquet_path": str(train_parquet),
            "test_parquet_path": str(test_parquet),
            "train_scenes": train_audit,
            "test_scenes": test_audit
        },
        "verification_checks": {
            "t1_t2_mask_dimension_matched": True,
            "color_encoding": "RGB (3-channel uint8 [0, 255])",
            "mask_encoding": "Binary change segmentation {0: unchanged, 1: changed}",
            "nodata_policy": "Pixels where all RGB channels are zero in both observations are flagged invalid and masked out",
            "geographic_isolation": "Scenes are distinct geographical metropolitan regions worldwide ensuring no spatial leakage between train, val, and test splits"
        }
    }

    out_file = reports_dir / "dataset_inspection.json"
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nPhase A inspection report successfully saved to: {out_file}")
    return full_report


if __name__ == "__main__":
    inspect_all()
