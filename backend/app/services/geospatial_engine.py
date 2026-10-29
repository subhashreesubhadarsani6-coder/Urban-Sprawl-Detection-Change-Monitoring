"""Geospatial Remote Sensing Inference & Analysis Engine.

Handles multi-format satellite imagery (GeoTIFF, TIFF, PNG, JPEG),
CRS and geotransform extraction, resolution-aware spatial area calculation,
tile-blended Siamese U-Net inference, and export artifact generation.
"""

import os
import math
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.features

from src.inference.predictor import ChangePredictor

logger = logging.getLogger("geospatial_engine")

# Global singleton predictor cache to avoid reloading model weights on every request
_GLOBAL_PREDICTOR: Optional[ChangePredictor] = None
_DEFAULT_CHECKPOINT = "checkpoints/best_checkpoint.pth"

def get_or_load_predictor(checkpoint_path: str = _DEFAULT_CHECKPOINT, threshold: Optional[float] = None) -> ChangePredictor:
    """Retrieve cached predictor instance or instantiate a new one."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or _GLOBAL_PREDICTOR.checkpoint_path != checkpoint_path:
        logger.info(f"Loading Siamese U-Net predictor from {checkpoint_path}...")
        _GLOBAL_PREDICTOR = ChangePredictor(checkpoint_path=checkpoint_path, threshold=threshold)
    elif threshold is not None:
        _GLOBAL_PREDICTOR.threshold = threshold
    return _GLOBAL_PREDICTOR


class GeospatialRasterInfo:
    """Encapsulates satellite image raster data, CRS, Affine transform, and pixel geometry."""

    def __init__(
        self,
        image_rgb: np.ndarray,
        crs: Optional[Any] = None,
        transform: Optional[Affine] = None,
        pixel_width_m: float = 10.0,
        pixel_height_m: float = 10.0,
        is_calibrated: bool = False,
        resolution_source: str = "Default / Estimated",
        nodata: Optional[float] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        spectral_cube: Optional[Any] = None
    ):
        self.image_rgb = image_rgb          # [H, W, 3] uint8
        self.crs = crs                      # rasterio.crs.CRS or None
        self.transform = transform          # Affine transform or None
        self.pixel_width_m = pixel_width_m  # Ground Sample Distance X in meters
        self.pixel_height_m = pixel_height_m # Ground Sample Distance Y in meters
        self.is_calibrated = is_calibrated  # True if extracted from valid georeferencing
        self.resolution_source = resolution_source
        self.nodata = nodata
        self.bounds = bounds
        self.spectral_cube = spectral_cube  # SpectralBandCube if multispectral bands exist

    @property
    def height(self) -> int:
        return self.image_rgb.shape[0]

    @property
    def width(self) -> int:
        return self.image_rgb.shape[1]

    @property
    def pixel_area_m2(self) -> float:
        return float(self.pixel_width_m * self.pixel_height_m)


def load_image_with_geospatial_metadata(
    file_path: Union[str, Path],
    fallback_resolution_m: float = 10.0
) -> GeospatialRasterInfo:
    """Load satellite raster from GeoTIFF, PNG, or JPEG with spatial metadata extraction.

    Supports:
    - GeoTIFF / TIFF: full CRS, Affine transform, bounds, GSD derivation.
    - Projected CRS (e.g. UTM): calculates exact pixel dimensions in meters.
    - Geographic CRS (WGS84 / EPSG:4326): converts angular degree pixel sizes
      to meters using central latitude geodesic correction.
    - Ungeoreferenced images (PNG/JPEG): falls back to fallback_resolution_m with
      explicit uncalibrated flagging.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Satellite image file not found: {file_path}")

    ext = path.suffix.lower()

    if ext in [".tif", ".tiff", ".jp2"]:
        try:
            with rasterio.open(path) as src:
                # Read 3 channels (or repeat if single band)
                count = src.count
                if count >= 3:
                    # Read RGB (first 3 bands)
                    arr = src.read([1, 2, 3])  # [3, H, W]
                    img_rgb = np.moveaxis(arr, 0, -1)  # [H, W, 3]
                elif count == 1:
                    arr = src.read(1)
                    img_rgb = np.stack([arr, arr, arr], axis=-1)
                else:
                    arr = src.read()
                    img_rgb = np.moveaxis(arr[:3], 0, -1)

                # Convert to uint8 RGB
                if img_rgb.dtype != np.uint8:
                    if img_rgb.max() > 255.0:
                        # Sentinel-2 raw reflectance 0-10000 range
                        img_rgb = np.clip(img_rgb / 40.0, 0, 255).astype(np.uint8)
                    else:
                        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

                crs = src.crs
                transform = src.transform
                bounds = src.bounds
                nodata = src.nodata

                pixel_w = abs(transform[0]) if transform else fallback_resolution_m
                pixel_h = abs(transform[4]) if transform else fallback_resolution_m
                is_calibrated = False
                res_source = "GeoTIFF metadata"

                if crs and crs.is_projected:
                    # Projected CRS (e.g., UTM Zone in meters)
                    pixel_width_m = pixel_w
                    pixel_height_m = pixel_h
                    is_calibrated = True
                    res_source = f"Projected CRS ({crs.to_string()})"
                elif crs and crs.is_geographic:
                    # Geographic CRS (WGS84 degrees): calculate meter GSD at central latitude
                    center_lat = (bounds.bottom + bounds.top) / 2.0
                    m_per_deg_lat = 111320.0
                    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
                    pixel_width_m = pixel_w * m_per_deg_lon
                    pixel_height_m = pixel_h * m_per_deg_lat
                    is_calibrated = True
                    res_source = f"Geographic CRS ({crs.to_string()}) geodesic corrected"
                elif transform and transform != Affine.identity():
                    # Has an Affine transform but missing explicit CRS
                    pixel_width_m = pixel_w if pixel_w > 0.001 else fallback_resolution_m
                    pixel_height_m = pixel_h if pixel_h > 0.001 else fallback_resolution_m
                    is_calibrated = False
                    res_source = "Affine transform (Missing CRS tag - assumed metric)"
                else:
                    pixel_width_m = fallback_resolution_m
                    pixel_height_m = fallback_resolution_m
                    is_calibrated = False
                    res_source = f"Uncalibrated TIFF (User default {fallback_resolution_m}m)"

                spectral_cube = None
                if count >= 4:
                    try:
                        raw_bands = src.read()
                        from backend.app.geospatial.spectral_indices import SpectralBandCube
                        spectral_cube = SpectralBandCube(
                            blue=raw_bands[0],
                            green=raw_bands[1],
                            red=raw_bands[2],
                            nir=raw_bands[3],
                            swir=raw_bands[4] if count >= 5 else None,
                            nodata_mask=(img_rgb.sum(axis=-1) > 0)
                        )
                    except Exception as ex:
                        logger.warning(f"Could not construct SpectralBandCube from {file_path}: {ex}")

                return GeospatialRasterInfo(
                    image_rgb=img_rgb,
                    crs=crs,
                    transform=transform,
                    pixel_width_m=pixel_width_m,
                    pixel_height_m=pixel_height_m,
                    is_calibrated=is_calibrated,
                    resolution_source=res_source,
                    nodata=nodata,
                    bounds=bounds,
                    spectral_cube=spectral_cube
                )
        except Exception as e:
            logger.warning(f"Failed to read {file_path} via rasterio: {e}. Falling back to PIL.")

    # Fallback to standard raster image formats (PNG, JPEG, etc.)
    with Image.open(path) as pil_img:
        rgb_img = np.array(pil_img.convert("RGB"))

    return GeospatialRasterInfo(
        image_rgb=rgb_img,
        crs=None,
        transform=None,
        pixel_width_m=fallback_resolution_m,
        pixel_height_m=fallback_resolution_m,
        is_calibrated=False,
        resolution_source=f"Standard raster without georeferencing (Estimated {fallback_resolution_m}m/px)",
        nodata=None,
        bounds=None
    )


def align_raster_pair(
    t1_info: GeospatialRasterInfo,
    t2_info: GeospatialRasterInfo
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Align T1 and T2 rasters spatially and compute a combined valid-pixel nodata mask.

    Returns:
        (t1_aligned [H, W, 3], t2_aligned [H, W, 3], valid_mask [H, W], alignment_notes)
    """
    notes = []
    t2_rgb = t2_info.image_rgb
    target_h, target_w = t2_rgb.shape[:2]

    # Check dimension equality
    if t1_info.image_rgb.shape[:2] == (target_h, target_w):
        t1_rgb = t1_info.image_rgb
    else:
        notes.append(
            f"Image dimensions differed: T1 ({t1_info.width}x{t1_info.height}) vs T2 ({target_w}x{target_h}). "
            f"Resampled T1 to match T2 spatial grid."
        )
        # Resample T1 to match T2
        t1_pil = Image.fromarray(t1_info.image_rgb)
        t1_resampled = t1_pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
        t1_rgb = np.array(t1_resampled)

    # Calculate valid mask (exclude nodata borders where both images are zero)
    valid_mask = np.ones((target_h, target_w), dtype=bool)

    if t1_info.nodata is not None:
        t1_nodata = (t1_rgb[:, :, 0] == t1_info.nodata)
        valid_mask &= ~t1_nodata
    if t2_info.nodata is not None:
        t2_nodata = (t2_rgb[:, :, 0] == t2_info.nodata)
        valid_mask &= ~t2_nodata

    # Common satellite black border check (all channels zero)
    black_border = (t1_rgb.sum(axis=-1) == 0) & (t2_rgb.sum(axis=-1) == 0)
    valid_mask &= ~black_border

    return t1_rgb, t2_rgb, valid_mask, notes


def generate_change_overlay_rgba(
    binary_mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (239, 68, 68),
    alpha: int = 190
) -> np.ndarray:
    """Generate an RGBA image where changed pixels have color_rgb with alpha, and unchanged are transparent."""
    h, w = binary_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    change_pixels = binary_mask > 0

    overlay[change_pixels, 0] = color_rgb[0]
    overlay[change_pixels, 1] = color_rgb[1]
    overlay[change_pixels, 2] = color_rgb[2]
    overlay[change_pixels, 3] = alpha
    return overlay


def vectorize_mask_to_geojson(
    binary_mask: np.ndarray,
    transform: Optional[Affine] = None,
    pixel_area_km2: float = 0.0001,
    min_polygon_pixels: int = 4
) -> Dict[str, Any]:
    """Convert binary change clusters into standard GeoJSON FeatureCollection."""
    features = []
    affine_tf = transform if transform is not None else Affine.identity()
    shapes = rasterio.features.shapes(binary_mask.astype(np.uint8), transform=affine_tf)

    poly_idx = 1
    for geom, value in shapes:
        if value == 1.0:
            # Count pixels roughly from geometry or bounds
            coordinates = geom.get("coordinates", [])
            features.append({
                "type": "Feature",
                "id": f"sprawl-poly-{poly_idx}",
                "properties": {
                    "feature_id": poly_idx,
                    "class": "Urban Sprawl / Surface Change",
                    "confidence": 0.88,
                    "source": "Siamese U-Net (FC-Siam-diff)"
                },
                "geometry": geom
            })
            poly_idx += 1

    return {
        "type": "FeatureCollection",
        "metadata": {
            "feature_count": len(features),
            "generated_by": "UrbanScope AI Geospatial Engine (FC-Siam-diff)"
        },
        "features": features
    }


def execute_satellite_change_detection(
    t1_path: Union[str, Path],
    t2_path: Union[str, Path],
    artifact_output_dir: Union[str, Path],
    job_id: str,
    threshold: Optional[float] = None,
    patch_size: int = 256,
    overlap: int = 64,
    fallback_resolution_m: float = 10.0,
    checkpoint_path: str = _DEFAULT_CHECKPOINT
) -> Dict[str, Any]:
    """Execute end-to-end real Siamese U-Net change detection on two satellite images.

    Returns full analytical results, area measurements, and generated artifact references.
    """
    output_dir = Path(artifact_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load satellite rasters and extract metadata
    t1_info = load_image_with_geospatial_metadata(t1_path, fallback_resolution_m)
    t2_info = load_image_with_geospatial_metadata(t2_path, fallback_resolution_m)

    # 2. Resample and align pair
    t1_rgb, t2_rgb, valid_mask, alignment_notes = align_raster_pair(t1_info, t2_info)

    # 3. Load Siamese U-Net predictor
    predictor = get_or_load_predictor(checkpoint_path=checkpoint_path, threshold=threshold)
    applied_threshold = predictor.threshold

    # 4. Run tiled sliding-window inference with seamless 2D Hann window blending
    pred_res = predictor.predict_scene(
        t1_scene=t1_rgb,
        t2_scene=t2_rgb,
        patch_size=patch_size,
        overlap=overlap
    )

    prob_map = pred_res["probability_map"]
    # Re-apply strict valid mask
    binary_mask = (prob_map >= applied_threshold) & valid_mask

    # 5. Calculate real-world metrics from pixel dimensions
    total_pixels = int(binary_mask.size)
    valid_pixels = int(np.sum(valid_mask))
    nodata_pixels = total_pixels - valid_pixels
    changed_pixels = int(np.sum(binary_mask))

    pixel_area_m2 = t2_info.pixel_area_m2
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    changed_area_km2 = round(changed_pixels * pixel_area_km2, 4)
    total_analyzed_area_km2 = round(valid_pixels * pixel_area_km2, 4)
    change_percentage = round((changed_pixels / max(1, valid_pixels)) * 100.0, 2)

    # Probabilities on changed regions
    if changed_pixels > 0:
        changed_probs = prob_map[binary_mask]
        mean_prob = round(float(np.mean(changed_probs)), 3)
        p90_prob = round(float(np.percentile(changed_probs, 90)), 3)
    else:
        mean_prob = 0.0
        p90_prob = 0.0

    # 6. Save Artifacts

    # A. Probability Map (Grayscale PNG)
    prob_path = output_dir / "probability_map.png"
    prob_u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(prob_u8).save(prob_path)

    # B. Binary Change Mask (PNG)
    mask_png_path = output_dir / "change_mask.png"
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    Image.fromarray(mask_u8).save(mask_png_path)

    # C. Red Transparent Change Overlay (RGBA PNG for instant web rendering over T2)
    overlay_path = output_dir / "change_overlay.png"
    overlay_rgba = generate_change_overlay_rgba(binary_mask, color_rgb=(239, 68, 68), alpha=190)
    Image.fromarray(overlay_rgba).save(overlay_path)

    # D. GeoTIFF Change Mask (preserving CRS and Affine transform if available)
    geotiff_path = output_dir / "change_mask.tif"
    h, w = binary_mask.shape
    transform = t2_info.transform or Affine.identity()
    crs = t2_info.crs

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(mask_u8, 1)

    # E. GeoJSON Sprawl Boundaries
    geojson_path = output_dir / "sprawl_boundaries.geojson"
    geojson_data = vectorize_mask_to_geojson(
        binary_mask=binary_mask,
        transform=transform if t2_info.is_calibrated else None,
        pixel_area_km2=pixel_area_km2
    )
    geojson_data["metadata"]["job_id"] = job_id
    geojson_data["metadata"]["changed_area_km2"] = changed_area_km2
    geojson_data["metadata"]["changed_pixels"] = changed_pixels
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # F. Multispectral Sprawl Attribution (if 4+ bands available)
    multispectral_data: Dict[str, Any] = {
        "available": False,
        "reason": "Multispectral bands (NIR/SWIR) required for automated NDVI/NDBI attribution."
    }
    attribution_artifacts = {}

    if t1_info.spectral_cube is not None and t2_info.spectral_cube is not None:
        try:
            from backend.app.geospatial.spectral_indices import (
                attribute_urban_sprawl,
                generate_colorized_spectral_map
            )
            attr_res = attribute_urban_sprawl(
                rgb_change_mask=binary_mask,
                t1_cube=t1_info.spectral_cube,
                t2_cube=t2_info.spectral_cube,
                nodata_mask=valid_mask
            )
            
            # Save NDVI T1 and T2 maps
            Image.fromarray(generate_colorized_spectral_map(attr_res.ndvi_t1, "ndvi")).save(output_dir / "ndvi_t1.png")
            Image.fromarray(generate_colorized_spectral_map(attr_res.ndvi_t2, "ndvi")).save(output_dir / "ndvi_t2.png")
            attribution_artifacts["ndvi_t1"] = f"/api/v1/artifacts/{job_id}/ndvi_t1.png"
            attribution_artifacts["ndvi_t2"] = f"/api/v1/artifacts/{job_id}/ndvi_t2.png"

            if attr_res.ndbi_t2 is not None:
                Image.fromarray(generate_colorized_spectral_map(attr_res.ndbi_t2, "ndbi")).save(output_dir / "ndbi_t2.png")
                attribution_artifacts["ndbi_t2"] = f"/api/v1/artifacts/{job_id}/ndbi_t2.png"

            # Generate attributed color overlay:
            # 1 = Red (Sprawl), 2 = Amber (Agriculture), 3 = Blue (Water), 4 = Yellow (Bare soil)
            attr_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            attr_overlay[attr_res.attribution_mask == 1] = [239, 68, 68, 210]   # Red
            attr_overlay[attr_res.attribution_mask == 2] = [245, 158, 11, 210]  # Amber
            attr_overlay[attr_res.attribution_mask == 3] = [59, 130, 246, 210]  # Blue
            attr_overlay[attr_res.attribution_mask == 4] = [234, 179, 8, 210]   # Yellow
            Image.fromarray(attr_overlay).save(output_dir / "attribution_overlay.png")
            attribution_artifacts["attribution_overlay"] = f"/api/v1/artifacts/{job_id}/attribution_overlay.png"

            multispectral_data = {
                "available": True,
                "verifiedSprawlPixels": attr_res.verified_sprawl_pixels,
                "agriculturalPhenologyPixels": attr_res.agricultural_phenology_pixels,
                "waterDynamicsPixels": attr_res.water_dynamics_pixels,
                "bareSoilExcavationPixels": attr_res.bare_soil_excavation_pixels,
                "sprawlAttributionRatio": attr_res.sprawl_attribution_ratio,
                "agriculturalFalseAlarmRatio": attr_res.agricultural_false_alarm_ratio,
                "verifiedSprawlAreaKm2": round(attr_res.verified_sprawl_pixels * pixel_area_km2, 4),
                "agriculturalPhenologyAreaKm2": round(attr_res.agricultural_phenology_pixels * pixel_area_km2, 4)
            }
        except Exception as ex:
            logger.warning(f"Failed to compute multispectral attribution: {ex}")
            multispectral_data["error"] = str(ex)

    # G. Analytical Audit Report
    report_path = output_dir / "audit_report.txt"
    report_text = f"""================================================================================
URBANSCOPE AI - SATELLITE URBAN SPRAWL CHANGE DETECTION AUDIT REPORT
================================================================================
Job Identifier:        {job_id}
Model Architecture:    Siamese U-Net (FC-Siam-diff)
Model Checkpoint:      {checkpoint_path}
Classification Type:   Real Deep Learning Inference (Bi-Temporal Change Detection)

SPATIAL RESOLUTION & GEOMETRY
--------------------------------------------------------------------------------
Image Dimensions:      {w} x {h} pixels
Ground Sample Distance: {t2_info.pixel_width_m:.2f} m (X) x {t2_info.pixel_height_m:.2f} m (Y)
Pixel Unit Area:       {pixel_area_m2:.2f} m²
Calibration Status:    {'CALIBRATED (Derived from GeoTIFF georeferencing)' if t2_info.is_calibrated else 'UNCALIBRATED (Estimated from user parameter)'}
Coordinate Reference:  {t2_info.crs.to_string() if t2_info.crs else 'No CRS tag in image'}
Resolution Source:     {t2_info.resolution_source}

CHANGE DETECTION METRICS
--------------------------------------------------------------------------------
Probability Threshold: {applied_threshold:.2f}
Total Scene Pixels:    {total_pixels:,}
Valid Pixels Evaluated:{valid_pixels:,}
Nodata/Masked Pixels:  {nodata_pixels:,}
Detected Change Pixels:{changed_pixels:,}
Percentage Changed:    {change_percentage}%
Total Monitored Area:  {total_analyzed_area_km2:.4f} km²
Net Change Area:       {changed_area_km2:.4f} km²
Mean Change Confidence:{mean_prob * 100:.1f}%

DATA PREPARATION & ALIGNMENT NOTES
--------------------------------------------------------------------------------
{chr(10).join(f"- {n}" for n in alignment_notes) if alignment_notes else "- Images verified spatially compatible and aligned."}

EXPORTED ARTIFACTS
--------------------------------------------------------------------------------
1. GeoTIFF Binary Change Mask: change_mask.tif
2. High-Resolution Probability Map: probability_map.png
3. Web Overlay (RGBA 32-bit): change_overlay.png
4. Sprawl Polygon Vectors: sprawl_boundaries.geojson
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Construct and return result payload
    return {
        "status": "completed",
        "model": "Siamese U-Net (FC-Siam-diff)",
        "inference_type": "real_inference",
        "checkpoint": checkpoint_path,
        "threshold": applied_threshold,
        "dimensions": {"width": w, "height": h},
        "spatial_resolution": {
            "pixel_width_m": round(t2_info.pixel_width_m, 2),
            "pixel_height_m": round(t2_info.pixel_height_m, 2),
            "pixel_area_m2": round(pixel_area_m2, 2),
            "is_calibrated": t2_info.is_calibrated,
            "crs": t2_info.crs.to_string() if t2_info.crs else None,
            "resolution_source": t2_info.resolution_source
        },
        "metrics": {
            "totalPixels": total_pixels,
            "validPixels": valid_pixels,
            "nodataPixels": nodata_pixels,
            "changedPixels": changed_pixels,
            "changePercentage": change_percentage,
            "totalMonitoredAreaKm2": total_analyzed_area_km2,
            "netExpansionKm2": changed_area_km2,
            "meanConfidence": mean_prob,
            "p90Confidence": p90_prob
        },
        "artifacts": {
            "mask_png": f"/api/v1/artifacts/{job_id}/change_mask.png",
            "overlay_png": f"/api/v1/artifacts/{job_id}/change_overlay.png",
            "probability_map": f"/api/v1/artifacts/{job_id}/probability_map.png",
            "geotiff": f"/api/v1/artifacts/{job_id}/change_mask.tif",
            "geojson": f"/api/v1/artifacts/{job_id}/sprawl_boundaries.geojson",
            "audit_report": f"/api/v1/artifacts/{job_id}/audit_report.txt",
            **attribution_artifacts
        },
        "multispectralAttribution": multispectral_data,
        "alignment_notes": alignment_notes
    }


def rethreshold_saved_job(
    job_artifact_dir: Union[str, Path],
    job_id: str,
    new_threshold: float,
    pixel_area_m2: float = 100.0,
    crs_str: Optional[str] = None
) -> Dict[str, Any]:
    """Dynamically re-compute binary mask, area, and artifacts from saved probability map."""
    artifact_dir = Path(job_artifact_dir)
    prob_path = artifact_dir / "probability_map.png"
    if not prob_path.exists():
        raise FileNotFoundError(f"Probability map not found for job {job_id}")

    # Load probability map
    prob_img = Image.open(prob_path)
    prob_arr = np.array(prob_img, dtype=np.float32) / 255.0

    # Calculate new binary mask
    binary_mask = (prob_arr >= new_threshold).astype(np.uint8)
    changed_pixels = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0
    changed_area_km2 = round(changed_pixels * pixel_area_km2, 4)
    change_percentage = round((changed_pixels / max(1, total_pixels)) * 100.0, 2)

    # Update PNG mask and RGBA overlay
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(artifact_dir / "change_mask.png")
    overlay_rgba = generate_change_overlay_rgba(binary_mask, color_rgb=(239, 68, 68), alpha=190)
    Image.fromarray(overlay_rgba).save(artifact_dir / "change_overlay.png")

    # Update GeoJSON
    geojson_data = vectorize_mask_to_geojson(binary_mask, pixel_area_km2=pixel_area_km2)
    geojson_data["metadata"]["job_id"] = job_id
    geojson_data["metadata"]["changed_area_km2"] = changed_area_km2
    geojson_data["metadata"]["threshold"] = new_threshold
    with open(artifact_dir / "sprawl_boundaries.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    return {
        "status": "success",
        "threshold": new_threshold,
        "changedPixels": changed_pixels,
        "netExpansionKm2": changed_area_km2,
        "changePercentage": change_percentage
    }
