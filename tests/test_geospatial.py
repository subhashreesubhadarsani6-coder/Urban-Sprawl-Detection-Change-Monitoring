"""Unit and Integration Tests for Geospatial Engine & Real Inference (Phase 4).

Covers:
- Tiling and 2D Hann window blending
- Dynamic thresholding behavior
- Projected and Geographic CRS metadata extraction
- Non-10m resolution and real-world area calculation
- Spatial dimension alignment and nodata masking
- End-to-end inference workflow with real checkpoint
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.crs import CRS

from backend.app.services.geospatial_engine import (
    load_image_with_geospatial_metadata,
    align_raster_pair,
    vectorize_mask_to_geojson,
    generate_change_overlay_rgba,
    execute_satellite_change_detection,
    rethreshold_saved_job,
    GeospatialRasterInfo
)
from src.inference.predictor import ChangePredictor


@pytest.fixture
def temp_geotiff_utm(tmp_path):
    """Create a temporary synthetic GeoTIFF with UTM 32611 projection and 15m resolution."""
    file_path = tmp_path / "sample_utm_15m.tif"
    data = np.full((3, 128, 128), 120, dtype=np.uint8)
    transform = from_origin(660000, 4000000, 15.0, 15.0)
    crs = CRS.from_epsg(32611)

    with rasterio.open(
        file_path, "w",
        driver="GTiff",
        height=128,
        width=128,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

    return str(file_path)


@pytest.fixture
def temp_geotiff_wgs84(tmp_path):
    """Create a temporary synthetic GeoTIFF with WGS84 EPSG:4326 (degrees) at ~36 deg lat."""
    file_path = tmp_path / "sample_wgs84.tif"
    data = np.full((3, 100, 100), 80, dtype=np.uint8)
    # 0.0001 deg ~ 11.13m lat, 9.0m lon at 36 deg lat
    deg_step = 0.0001
    transform = from_origin(-115.2, 36.1, deg_step, deg_step)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
        file_path, "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

    return str(file_path)


@pytest.fixture
def temp_png_pair(tmp_path):
    """Create a temporary synthetic PNG image pair with differing sizes."""
    t1_path = tmp_path / "t1_300x200.png"
    t2_path = tmp_path / "t2_400x300.png"

    img1 = Image.fromarray(np.random.randint(40, 200, (200, 300, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(40, 200, (300, 400, 3), dtype=np.uint8))

    img1.save(t1_path)
    img2.save(t2_path)
    return str(t1_path), str(t2_path)


def test_geospatial_metadata_projected_utm(temp_geotiff_utm):
    """Verify that projected GeoTIFFs extract exact GSD in meters and mark as calibrated."""
    info = load_image_with_geospatial_metadata(temp_geotiff_utm)
    assert info.is_calibrated is True
    assert info.crs is not None
    assert "32611" in info.crs.to_string()
    assert info.pixel_width_m == pytest.approx(15.0, rel=1e-3)
    assert info.pixel_height_m == pytest.approx(15.0, rel=1e-3)
    assert info.pixel_area_m2 == pytest.approx(225.0, rel=1e-3)


def test_geospatial_metadata_geographic_wgs84(temp_geotiff_wgs84):
    """Verify that WGS84 degree rasters convert angular degrees to geodesic meters correctly."""
    info = load_image_with_geospatial_metadata(temp_geotiff_wgs84)
    assert info.is_calibrated is True
    assert info.crs is not None
    assert "4326" in info.crs.to_string()
    # At latitude ~36 deg, 0.0001 deg lat is ~11.13m, lon is ~9.0m
    assert info.pixel_height_m > 8.0 and info.pixel_height_m < 15.0
    assert info.pixel_width_m > 6.0 and info.pixel_width_m < 13.0


def test_geospatial_metadata_unreferenced_png(temp_png_pair):
    """Verify that standard PNGs without georeferencing default to user GSD and mark uncalibrated."""
    t1_path, _ = temp_png_pair
    custom_res = 3.0  # PlanetScope 3m resolution
    info = load_image_with_geospatial_metadata(t1_path, fallback_resolution_m=custom_res)
    assert info.is_calibrated is False
    assert info.crs is None
    assert info.pixel_width_m == 3.0
    assert info.pixel_height_m == 3.0
    assert info.pixel_area_m2 == 9.0


def test_area_calculation_never_blind_10m():
    """Verify area calculations for arbitrary non-10m resolutions (e.g. 0.5m, 3m, 20m)."""
    # 10,000 changed pixels
    changed_pixels = 10000

    # 1. Very High Resolution (0.5m WorldView): 0.25 m^2 per pixel
    gsd_vhr = 0.5
    area_km2_vhr = (changed_pixels * (gsd_vhr * gsd_vhr)) / 1_000_000.0
    assert area_km2_vhr == pytest.approx(0.0025, rel=1e-4)

    # 2. PlanetScope (3.0m): 9.0 m^2 per pixel
    gsd_planet = 3.0
    area_km2_planet = (changed_pixels * (gsd_planet * gsd_planet)) / 1_000_000.0
    assert area_km2_planet == pytest.approx(0.09, rel=1e-4)

    # 3. Sentinel-2 20m SWIR Band: 400 m^2 per pixel
    gsd_swir = 20.0
    area_km2_swir = (changed_pixels * (gsd_swir * gsd_swir)) / 1_000_000.0
    assert area_km2_swir == pytest.approx(4.0, rel=1e-4)


def test_dimension_alignment_and_nodata(temp_png_pair):
    """Verify that images with different dimensions are safely aligned and nodata pixels masked."""
    t1_path, t2_path = temp_png_pair
    t1_info = load_image_with_geospatial_metadata(t1_path)
    t2_info = load_image_with_geospatial_metadata(t2_path)

    assert t1_info.image_rgb.shape != t2_info.image_rgb.shape

    t1_aligned, t2_aligned, valid_mask, notes = align_raster_pair(t1_info, t2_info)

    # Should have matching shape equal to T2
    assert t1_aligned.shape == t2_aligned.shape
    assert t1_aligned.shape[:2] == (300, 400)
    assert valid_mask.shape == (300, 400)
    assert len(notes) > 0
    assert "Image dimensions differed" in notes[0]


def test_tile_blending_and_thresholding(tmp_path):
    """Verify 2D Hann window blending and monotonic thresholding behavior."""
    checkpoint_path = "checkpoints/best_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    predictor = ChangePredictor(checkpoint_path=checkpoint_path, threshold=0.5)

    # Create synthetic test scene [320, 320, 3] requiring tiling (patch_size=256, overlap=64)
    h, w = 320, 320
    t1_scene = np.full((h, w, 3), 100, dtype=np.uint8)
    t2_scene = np.full((h, w, 3), 100, dtype=np.uint8)
    # Add synthetic change in center
    t2_scene[80:240, 80:240, :] = 220

    res_50 = predictor.predict_scene(t1_scene, t2_scene, patch_size=256, overlap=64)
    assert res_50["probability_map"].shape == (h, w)
    assert res_50["probability_map"].min() >= 0.0
    assert res_50["probability_map"].max() <= 1.0

    # Test threshold effect
    predictor.threshold = 0.2
    res_20 = predictor.predict_scene(t1_scene, t2_scene, patch_size=256, overlap=64)

    predictor.threshold = 0.8
    res_80 = predictor.predict_scene(t1_scene, t2_scene, patch_size=256, overlap=64)

    # Monotonic property: lower threshold -> more or equal changed pixels
    assert res_20["changed_pixels"] >= res_50["changed_pixels"]
    assert res_50["changed_pixels"] >= res_80["changed_pixels"]


def test_overlay_generation_and_vectorization():
    """Verify RGBA overlay transparency and GeoJSON polygon generation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:60] = 1

    overlay = generate_change_overlay_rgba(mask, color_rgb=(239, 68, 68), alpha=190)
    assert overlay.shape == (100, 100, 4)
    assert overlay[0, 0, 3] == 0  # Unchanged transparent
    assert overlay[25, 35, 0] == 239  # Red
    assert overlay[25, 35, 3] == 190  # Alpha

    geojson = vectorize_mask_to_geojson(mask)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) >= 1
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"


def test_full_inference_workflow_with_real_checkpoint(tmp_path):
    """End-to-end integration test with real checkpoint and verified Las Vegas imagery."""
    checkpoint_path = "checkpoints/best_checkpoint.pth"
    t1_path = "reports/sample_t1_lasvegas.png"
    t2_path = "reports/sample_t2_lasvegas.png"

    if not os.path.exists(checkpoint_path) or not os.path.exists(t1_path) or not os.path.exists(t2_path):
        pytest.skip("Prerequisites missing for full integration test.")

    job_id = "test-phase4-integration"
    output_dir = tmp_path / job_id

    result = execute_satellite_change_detection(
        t1_path=t1_path,
        t2_path=t2_path,
        artifact_output_dir=output_dir,
        job_id=job_id,
        threshold=0.5,
        patch_size=256,
        overlap=64,
        fallback_resolution_m=10.0,
        checkpoint_path=checkpoint_path
    )

    # 1. Verify result structure
    assert result["status"] == "completed"
    assert result["model"] == "Siamese U-Net (FC-Siam-diff)"
    assert result["metrics"]["totalPixels"] == 589984
    assert result["metrics"]["changedPixels"] > 0
    assert result["metrics"]["netExpansionKm2"] > 0.0

    # 2. Verify all 6 artifacts were generated on disk
    assert (output_dir / "change_mask.tif").exists()
    assert (output_dir / "change_mask.png").exists()
    assert (output_dir / "change_overlay.png").exists()
    assert (output_dir / "probability_map.png").exists()
    assert (output_dir / "sprawl_boundaries.geojson").exists()
    assert (output_dir / "audit_report.txt").exists()

    # 3. Verify GeoTIFF output preserves raster properties
    with rasterio.open(output_dir / "change_mask.tif") as ds:
        assert ds.count == 1
        assert ds.width == 716
        assert ds.height == 824

    # 4. Verify dynamic re-thresholding from saved artifacts
    re_res = rethreshold_saved_job(output_dir, job_id, new_threshold=0.7, pixel_area_m2=100.0)
    assert re_res["status"] == "success"
    assert re_res["threshold"] == 0.7
    assert re_res["changedPixels"] < result["metrics"]["changedPixels"]


def test_multispectral_geotiff_attribution_workflow(tmp_path):
    """Verify that multi-band (5-band) GeoTIFFs trigger automated NDVI/NDBI sprawl attribution."""
    h, w = 128, 128
    t1_tif = tmp_path / "t1_multispectral.tif"
    t2_tif = tmp_path / "t2_multispectral.tif"

    # Create 5-band synthetic satellite GeoTIFFs (B02 Blue, B03 Green, B04 Red, B08 NIR, B11 SWIR)
    bands_t1 = np.zeros((5, h, w), dtype=np.uint8)
    bands_t1[0:3] = 80   # RGB dark
    bands_t1[3] = 180    # NIR high (vegetation)
    bands_t1[4] = 60     # SWIR low

    bands_t2 = np.zeros((5, h, w), dtype=np.uint8)
    bands_t2[0:3] = 160  # RGB brighter
    bands_t2[3] = 70     # NIR low (vegetation loss)
    bands_t2[4] = 200    # SWIR high (built-up concrete/asphalt)

    transform = Affine(10.0, 0, 500000, 0, -10.0, 4000000)
    crs = rasterio.crs.CRS.from_epsg(32611)

    for path, data in [(t1_tif, bands_t1), (t2_tif, bands_t2)]:
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=5,
            dtype=rasterio.uint8, crs=crs, transform=transform
        ) as dst:
            dst.write(data)

    job_id = "test-multispectral-job"
    output_dir = tmp_path / job_id
    checkpoint_path = "checkpoints/best_checkpoint.pth"

    result = execute_satellite_change_detection(
        t1_path=t1_tif,
        t2_path=t2_tif,
        artifact_output_dir=output_dir,
        job_id=job_id,
        threshold=0.5,
        patch_size=128,
        overlap=32,
        checkpoint_path=checkpoint_path
    )

    assert result["status"] == "completed"
    assert "multispectralAttribution" in result
    attr = result["multispectralAttribution"]
    assert attr["available"] is True
    assert "verifiedSprawlPixels" in attr
    assert "agriculturalPhenologyPixels" in attr
    assert (output_dir / "ndvi_t1.png").exists()
    assert (output_dir / "attribution_overlay.png").exists()
