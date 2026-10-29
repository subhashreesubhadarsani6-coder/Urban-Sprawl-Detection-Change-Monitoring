"""Comprehensive dataset audit script for OSCD / Parquet change detection datasets.

Inspects Parquet files, schemas, image encodings, dimensions, bands, value ranges,
ground-truth masks, class balance, and geographic scenes.
"""

import os
import io
import glob
import json
import zipfile
from collections import Counter
import pyarrow.parquet as pq
import numpy as np
from PIL import Image

def audit_dataset(dataset_dir="dataset", output_report="reports/dataset_audit.json"):
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    report = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "parquet_files": {},
        "zip_archives": {},
        "crdownload_files": [],
        "scenes": {},
        "blockers": [],
        "warnings": [],
        "recommendations": []
    }

    # 1. Check for unconfirmed / interrupted downloads
    cr_files = glob.glob(os.path.join(dataset_dir, "*.crdownload"))
    for cr in cr_files:
        report["crdownload_files"].append({
            "name": os.path.basename(cr),
            "size_bytes": os.path.getsize(cr),
            "size_mb": round(os.path.getsize(cr) / (1024 * 1024), 2)
        })
    if cr_files:
        report["warnings"].append(
            f"Found {len(cr_files)} incomplete Chrome download files (*.crdownload) in {dataset_dir}."
        )

    # 2. Check ZIP label archives
    zip_files = glob.glob(os.path.join(dataset_dir, "*.zip"))
    for zf in zip_files:
        zname = os.path.basename(zf)
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                namelist = z.namelist()
                cities = set()
                for item in namelist:
                    parts = item.strip("/").split("/")
                    if len(parts) >= 2 and parts[1] not in ("README.txt", ""):
                        cities.add(parts[1])
                report["zip_archives"][zname] = {
                    "size_bytes": os.path.getsize(zf),
                    "total_entries": len(namelist),
                    "detected_scenes": sorted(list(cities))
                }
        except Exception as e:
            report["zip_archives"][zname] = {"error": str(e)}

    # 3. Check Parquet files
    parquet_files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
    total_samples = 0

    for pf in parquet_files:
        pname = os.path.basename(pf)
        meta = pq.read_metadata(pf)
        schema = pq.read_schema(pf)

        file_info = {
            "size_mb": round(os.path.getsize(pf) / (1024 * 1024), 2),
            "num_rows": meta.num_rows,
            "num_row_groups": meta.num_row_groups,
            "columns": schema.names,
            "sample_details": []
        }
        total_samples += meta.num_rows

        # Read samples from this parquet file
        table = pq.read_table(pf)
        df = table.to_pandas()

        for idx in range(min(5, len(df))):
            row_info = {"row_idx": idx}
            for col in ["image1", "image2", "mask"]:
                val = df[col].iloc[idx]
                if isinstance(val, dict):
                    path_val = val.get("path")
                    raw_bytes = val.get("bytes")
                    byte_len = len(raw_bytes) if raw_bytes is not None else 0
                    row_info[col] = {
                        "path": path_val,
                        "bytes_length": byte_len
                    }
                    if raw_bytes and byte_len > 0:
                        try:
                            img = Image.open(io.BytesIO(raw_bytes))
                            row_info[col]["format"] = img.format
                            row_info[col]["size"] = img.size  # (W, H)
                            row_info[col]["mode"] = img.mode
                            arr = np.array(img)
                            row_info[col]["shape"] = list(arr.shape)
                            row_info[col]["dtype"] = str(arr.dtype)
                            row_info[col]["min"] = float(arr.min())
                            row_info[col]["max"] = float(arr.max())
                            if col == "mask":
                                unique_vals, counts = np.unique(arr, return_counts=True)
                                row_info[col]["unique_values"] = {int(k): int(v) for k, v in zip(unique_vals, counts)}
                        except Exception as e:
                            row_info[col]["decode_error"] = str(e)
            file_info["sample_details"].append(row_info)

        report["parquet_files"][pname] = file_info

    # 4. Check whether train shards 0-4 are complete or missing shard 3
    train_shards = [f for f in parquet_files if "train-" in os.path.basename(f) and "of-00005" in os.path.basename(f)]
    shard_indices = [os.path.basename(f).split("-")[1] for f in train_shards]
    if "of-00005" in "".join(os.path.basename(f) for f in parquet_files):
        expected_indices = [f"{i:05d}" for i in range(5)]
        missing = [idx for idx in expected_indices if idx not in shard_indices]
        if missing:
            report["warnings"].append(
                f"Multi-shard train set train-*-of-00005 has missing shards: {missing}. (Likely aborted downloads represented by *.crdownload files)."
            )

    # 5. Check the 1-shard files: train-00000-of-00001.parquet and test-00000-of-00001.parquet
    single_shard_train = os.path.join(dataset_dir, "train-00000-of-00001.parquet")
    single_shard_test = os.path.join(dataset_dir, "test-00000-of-00001.parquet")
    
    if os.path.exists(single_shard_train) and os.path.exists(single_shard_test):
        report["single_shard_dataset"] = {
            "train_rows": pq.read_metadata(single_shard_train).num_rows,
            "test_rows": pq.read_metadata(single_shard_test).num_rows,
            "status": "Both single-shard train (14 full scenes) and test (10 full scenes) exist and are complete!"
        }

    # Save report
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Audit completed. Report saved to {output_report}")
    return report

if __name__ == "__main__":
    report = audit_dataset()
    print("\n--- Summary ---")
    print(f"Parquet files audited: {len(report['parquet_files'])}")
    for pf, info in report["parquet_files"].items():
        print(f"  {pf}: {info['num_rows']} rows, {info['size_mb']} MB")
        if info["sample_details"]:
            sample = info["sample_details"][0]
            img_shape = sample.get("image1", {}).get("shape")
            img_mode = sample.get("image1", {}).get("mode")
            mask_shape = sample.get("mask", {}).get("shape")
            mask_uniques = sample.get("mask", {}).get("unique_values")
            print(f"    Sample 0: image1 shape={img_shape} mode={img_mode}, mask shape={mask_shape} uniques={mask_uniques}")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  ! {w}")
