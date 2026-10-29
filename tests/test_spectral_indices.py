"""Unit and integration tests for multispectral remote sensing indices and sprawl attribution."""
import pytest
import numpy as np

from backend.app.geospatial.spectral_indices import (
    calculate_ndvi,
    calculate_ndbi,
    calculate_mndwi,
    safe_normalized_difference,
    SpectralBandCube,
    attribute_urban_sprawl,
    generate_colorized_spectral_map,
)


def test_safe_normalized_difference_zero_denominator():
    """Ensure safe division prevents NaN/inf on pure zero or near-zero inputs."""
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.zeros((10, 10), dtype=np.float32)
    result = safe_normalized_difference(a, b)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert np.all(result == 0.0)


def test_calculate_ndvi_bounds():
    """Verify NDVI values strictly fall within [-1.0, 1.0]."""
    # NIR high, Red low -> dense healthy canopy
    nir_veg = np.full((5, 5), 0.8, dtype=np.float32)
    red_veg = np.full((5, 5), 0.1, dtype=np.float32)
    ndvi_veg = calculate_ndvi(nir_veg, red_veg)
    assert np.all(ndvi_veg > 0.7)
    assert np.all(ndvi_veg <= 1.0)

    # Water: Red > NIR
    nir_water = np.full((5, 5), 0.05, dtype=np.float32)
    red_water = np.full((5, 5), 0.20, dtype=np.float32)
    ndvi_water = calculate_ndvi(nir_water, red_water)
    assert np.all(ndvi_water < 0.0)
    assert np.all(ndvi_water >= -1.0)


def test_calculate_ndbi_bounds():
    """Verify NDBI values correctly identify impervious surfaces."""
    # Built-up: SWIR > NIR
    swir_urban = np.full((4, 4), 0.45, dtype=np.float32)
    nir_urban = np.full((4, 4), 0.25, dtype=np.float32)
    ndbi_urban = calculate_ndbi(swir_urban, nir_urban)
    assert np.all(ndbi_urban > 0.0)

    # Vegetation: NIR >> SWIR
    swir_veg = np.full((4, 4), 0.10, dtype=np.float32)
    nir_veg = np.full((4, 4), 0.60, dtype=np.float32)
    ndbi_veg = calculate_ndbi(swir_veg, nir_veg)
    assert np.all(ndbi_veg < -0.6)


def test_spectral_cube_validation_and_scaling():
    """Verify cube input validation and automated raw DN (0-10000) scaling."""
    # Raw Sentinel-2 0-10000 range
    raw_b = np.full((8, 8), 1500, dtype=np.float32)
    raw_g = np.full((8, 8), 1800, dtype=np.float32)
    raw_r = np.full((8, 8), 2000, dtype=np.float32)
    raw_nir = np.full((8, 8), 5000, dtype=np.float32)
    raw_swir = np.full((8, 8), 3000, dtype=np.float32)

    cube = SpectralBandCube(
        blue=raw_b, green=raw_g, red=raw_r, nir=raw_nir, swir=raw_swir
    )
    # Scaled down to [0.0, 1.0]
    assert np.isclose(cube.nir.max(), 0.50)
    assert np.isclose(cube.red.max(), 0.20)

    # Mismatched dimension raises error
    with pytest.raises(ValueError, match="Band dimension mismatch"):
        SpectralBandCube(
            blue=raw_b, green=raw_g, red=raw_r, nir=np.zeros((10, 10))
        )


def test_multispectral_sprawl_attribution_separation():
    """Verify that agriculture/harvest is distinguished from concrete/urban expansion."""
    h, w = 10, 10
    # Both pixels identified as "change" by RGB Siamese U-Net
    rgb_change = np.zeros((h, w), dtype=np.uint8)
    rgb_change[2, 2] = 1  # Urban sprawl scenario
    rgb_change[7, 7] = 1  # Agricultural harvest scenario

    # Pixel (2, 2): Starts as natural grass/soil, becomes concrete/subdivision
    # T1: moderate veg (NIR=0.4, Red=0.15, SWIR=0.15) -> NDVI~0.45, NDBI~0.0
    # T2: concrete building (NIR=0.2, Red=0.25, SWIR=0.40) -> NDVI~-0.11, NDBI~+0.33
    t1_b = np.full((h, w), 0.1, dtype=np.float32)
    t1_g = np.full((h, w), 0.15, dtype=np.float32)
    t1_r = np.full((h, w), 0.15, dtype=np.float32)
    t1_nir = np.full((h, w), 0.40, dtype=np.float32)
    t1_swir = np.full((h, w), 0.15, dtype=np.float32)

    t2_b = np.full((h, w), 0.1, dtype=np.float32)
    t2_g = np.full((h, w), 0.15, dtype=np.float32)
    t2_r = np.full((h, w), 0.15, dtype=np.float32)
    t2_nir = np.full((h, w), 0.40, dtype=np.float32)
    t2_swir = np.full((h, w), 0.15, dtype=np.float32)

    # Modify (2, 2) in T2 to be concrete
    t2_r[2, 2] = 0.25
    t2_nir[2, 2] = 0.20
    t2_swir[2, 2] = 0.40

    # Modify (7, 7) to be crop harvest (was dense green, harvested to bare soil with no concrete signature)
    # T1: Dense crop (NIR=0.7, Red=0.1) -> NDVI~0.75, NDBI~-0.6
    # T2: Bare soil harvest (NIR=0.25, Red=0.22, SWIR=0.20) -> NDVI~0.06, NDBI~-0.11 (negative NDBI!)
    t1_nir[7, 7] = 0.70
    t1_r[7, 7] = 0.10
    t1_swir[7, 7] = 0.15

    t2_nir[7, 7] = 0.25
    t2_r[7, 7] = 0.22
    t2_swir[7, 7] = 0.20

    cube1 = SpectralBandCube(blue=t1_b, green=t1_g, red=t1_r, nir=t1_nir, swir=t1_swir)
    cube2 = SpectralBandCube(blue=t2_b, green=t2_g, red=t2_r, nir=t2_nir, swir=t2_swir)

    res = attribute_urban_sprawl(rgb_change, cube1, cube2)

    # Pixel (2, 2) must be attributed to Class 1 (Verified Urban Sprawl)
    assert res.attribution_mask[2, 2] == 1, "Concrete expansion must be classified as Urban Sprawl"
    # Pixel (7, 7) must be attributed to Class 2 (Agricultural Phenology)
    assert res.attribution_mask[7, 7] == 2, "Crop harvesting must be separated into Agricultural Phenology"

    assert res.verified_sprawl_pixels == 1
    assert res.agricultural_phenology_pixels == 1
    assert res.raw_rgb_change_pixels == 2


def test_generate_colorized_spectral_map():
    """Verify that 2D spectral index maps convert to valid 3-channel RGB visuals."""
    sample_ndvi = np.linspace(-1.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    rgb_map = generate_colorized_spectral_map(sample_ndvi, color_scheme="ndvi")
    assert rgb_map.shape == (10, 10, 3)
    assert rgb_map.dtype == np.uint8
    assert rgb_map.min() >= 0
    assert rgb_map.max() <= 255
