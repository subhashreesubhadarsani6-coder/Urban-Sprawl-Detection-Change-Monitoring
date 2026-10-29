"""Standalone Inference Module for Bi-temporal Satellite Change Detection.

Accepts T1 (earlier) and T2 (later) satellite image paths (PNG, JPEG, TIFF, GeoTIFF),
loads trained Siamese U-Net checkpoint, executes inference, and generates:
1. Probability map (grayscale / colormapped PNG or GeoTIFF)
2. Binary change mask (0 = unchanged, 255 = changed)
3. Red change overlay blended with T2 observation
4. JSON prediction metadata (including pixel counts, area calculation if georeferenced)
Preserves geospatial metadata (CRS, Affine Transform) when processing GeoTIFF inputs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import torch

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from models.siam_unet import FCSiamDiff
from src.data.transforms import normalize_imagery


def load_image_with_metadata(image_path: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load image from path, handling RGB conversion and geospatial metadata if GeoTIFF."""
    meta = None
    if RASTERIO_AVAILABLE and (image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff")):
        try:
            with rasterio.open(image_path) as src:
                # Read first 3 channels (or single channel replicated)
                count = src.count
                if count >= 3:
                    img_data = src.read([1, 2, 3])  # [3, H, W]
                    img_rgb = np.transpose(img_data, (1, 2, 0))  # [H, W, 3]
                else:
                    ch1 = src.read(1)
                    img_rgb = np.stack([ch1, ch1, ch1], axis=-1)

                # Normalize to uint8 [0, 255] if 16-bit
                if img_rgb.dtype == np.uint16:
                    img_rgb = np.clip(img_rgb / 256.0, 0, 255).astype(np.uint8)
                elif img_rgb.dtype != np.uint8:
                    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

                meta = {
                    "crs": str(src.crs) if src.crs else None,
                    "transform": list(src.transform) if src.transform else None,
                    "width": src.width,
                    "height": src.height,
                    "pixel_size_x": abs(src.transform[0]) if src.transform else None,
                    "pixel_size_y": abs(src.transform[4]) if src.transform else None,
                    "is_projected": bool(src.crs.is_projected) if src.crs else False
                }
                return img_rgb, meta
        except Exception as e:
            print(f"Notice: rasterio load failed ({e}), falling back to PIL.")

    pil_img = Image.open(image_path).convert("RGB")
    arr = np.array(pil_img)
    return arr, None


def predict_change(
    t1_path: str,
    t2_path: str,
    checkpoint_path: str = "artifacts/checkpoints/best_model.pth",
    output_dir: str = "reports/predictions",
    threshold: float = 0.5,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """Execute change detection inference between T1 and T2 observations."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Running inference on device: {device}")

    # Fallback checkpoint check
    if not os.path.exists(checkpoint_path):
        if os.path.exists("checkpoints/best_checkpoint.pth"):
            checkpoint_path = "checkpoints/best_checkpoint.pth"
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FCSiamDiff(in_channels=3, base_channels=32, num_classes=1).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Load imagery
    img1, meta1 = load_image_with_metadata(t1_path)
    img2, meta2 = load_image_with_metadata(t2_path)

    # Validate spatial dimensions
    if img1.shape[:2] != img2.shape[:2]:
        print(f"Notice: Resizing T2 from {img2.shape[:2]} to match T1 {img1.shape[:2]}")
        h, w = img1.shape[:2]
        img2 = np.array(Image.fromarray(img2).resize((w, h), Image.Resampling.BILINEAR))

    h, w, c = img1.shape

    # Normalize inputs
    t1_norm = normalize_imagery(img1)
    t2_norm = normalize_imagery(img2)

    # Tiled inference with Hann window blending for arbitrary image dimensions
    patch_size = 256
    stride = 192

    prob_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    # 2D Hann window
    hann_1d = np.hanning(patch_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    hann_2d = np.maximum(hann_2d, 1e-4)

    # Sliding window tiles
    y_starts = list(range(0, max(1, h - patch_size + 1), stride))
    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)
    x_starts = list(range(0, max(1, w - patch_size + 1), stride))
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)

    with torch.no_grad():
        for top in y_starts:
            for left in x_starts:
                t1_crop = t1_norm[top:top + patch_size, left:left + patch_size]
                t2_crop = t2_norm[top:top + patch_size, left:left + patch_size]

                t1_t = torch.from_numpy(t1_crop.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                t2_t = torch.from_numpy(t2_crop.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                logits = model(t1_t, t2_t)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_accum[top:top + patch_size, left:left + patch_size] += prob * hann_2d
                weight_accum[top:top + patch_size, left:left + patch_size] += hann_2d

    weight_accum = np.maximum(weight_accum, 1e-5)
    prob_map = prob_accum / weight_accum
    prob_map = np.clip(prob_map, 0.0, 1.0)

    # Binary mask
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    # Changed pixels and area calculations
    total_pixels = int(h * w)
    changed_pixels = int(np.sum(binary_mask == 1))
    change_pct = round(100.0 * changed_pixels / max(1, total_pixels), 3)

    # Determine resolution and area
    geo_meta = meta1 or meta2
    area_km2 = None
    area_calculation_valid = False
    resolution_description = "Uncalibrated pixel grid (fallback 10m GSD used for estimation)"

    if geo_meta and geo_meta.get("pixel_size_x"):
        res_x = geo_meta["pixel_size_x"]
        res_y = geo_meta["pixel_size_y"]
        if geo_meta.get("is_projected"):
            # Projected CRS (meters)
            pixel_area_m2 = res_x * res_y
            area_km2 = round((changed_pixels * pixel_area_m2) / 1e6, 5)
            area_calculation_valid = True
            resolution_description = f"Calibrated {res_x:.1f}m x {res_y:.1f}m GSD via CRS: {geo_meta.get('crs')}"
        else:
            # Geographic (degrees) - approximate geodesic conversion
            lat_mid = 0.0  # approximate equator fallback
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mid))
            pixel_area_m2 = (res_x * m_per_deg_lon) * (res_y * m_per_deg_lat)
            area_km2 = round((changed_pixels * pixel_area_m2) / 1e6, 5)
            area_calculation_valid = True
            resolution_description = f"Calibrated geographic coordinates ({geo_meta.get('crs')})"
    else:
        # Fallback 10m assumption clearly labeled
        area_km2 = round((changed_pixels * 100.0) / 1e6, 5)
        area_calculation_valid = False

    # Red change overlay blended on T2
    overlay = img2.copy()
    red_color = np.array([239, 68, 68], dtype=np.uint8)
    alpha = 0.65
    mask_bool = binary_mask == 1
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + red_color * alpha).astype(np.uint8)

    # Save output artifacts
    prob_path = os.path.join(output_dir, "probability_map.png")
    mask_path = os.path.join(output_dir, "change_mask.png")
    overlay_path = os.path.join(output_dir, "change_overlay.png")
    meta_path = os.path.join(output_dir, "prediction_metadata.json")

    Image.fromarray((prob_map * 255).astype(np.uint8)).save(prob_path)
    Image.fromarray(binary_mask * 255).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    # Save GeoTIFF mask if geospatial metadata is available
    geotiff_path = None
    if RASTERIO_AVAILABLE and geo_meta and geo_meta.get("transform"):
        try:
            geotiff_path = os.path.join(output_dir, "change_mask.tif")
            with rasterio.open(
                geotiff_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="uint8",
                crs=geo_meta["crs"],
                transform=rasterio.Affine(*geo_meta["transform"][:6])
            ) as dst:
                dst.write(binary_mask, 1)
        except Exception as e:
            print(f"Notice: GeoTIFF export skipped ({e})")

    metadata = {
        "status": "SUCCESS",
        "t1_path": t1_path,
        "t2_path": t2_path,
        "checkpoint_used": checkpoint_path,
        "image_dimensions": {"height": h, "width": w, "channels": c},
        "applied_threshold": threshold,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": change_pct,
        "area_km2": area_km2,
        "area_calculation_valid": area_calculation_valid,
        "resolution_description": resolution_description,
        "artifacts": {
            "probability_map": prob_path,
            "change_mask": mask_path,
            "change_overlay": overlay_path,
            "change_mask_geotiff": geotiff_path,
            "prediction_metadata": meta_path
        }
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Inference Results ---")
    print(f"Scene Dimensions:    {w} x {h}")
    print(f"Changed Pixels:      {changed_pixels:,} / {total_pixels:,} ({change_pct}%)")
    print(f"Expansion Area:      {area_km2} km² ({resolution_description})")
    print(f"Artifacts saved in:  {output_dir}/")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese U-Net Satellite Change Inference.")
    parser.add_argument("--t1", type=str, required=True, help="Path to T1 observation")
    parser.add_argument("--t2", type=str, required=True, help="Path to T2 observation")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="reports/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict_change(
        t1_path=args.t1,
        t2_path=args.t2,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
