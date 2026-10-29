"""Multispectral Remote Sensing Indices & Urban Sprawl Attribution Engine.

Provides mathematically rigorous, numerically stable calculations for:
- NDVI (Normalized Difference Vegetation Index): (NIR - Red) / (NIR + Red)
- NDBI (Normalized Difference Built-up Index): (SWIR - NIR) / (SWIR + NIR)
- MNDWI (Modified Normalized Difference Water Index): (Green - SWIR) / (Green + SWIR)

Combines deep-learning bi-temporal change likelihood maps with multispectral
signatures to attribute detected change clusters into:
1. Verified Urban Sprawl (Built-up expansion, concrete/asphalt accumulation, impervious surface)
2. Agricultural Phenology (Crop harvest, vegetation green-up/senescence)
3. Water Body Dynamics (Reservoir changes, flood/drought fluctuations)
4. Earthwork / Bare Soil Excavation (Pre-construction land grading)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import numpy as np
from PIL import Image

logger = logging.getLogger("spectral_indices")


def safe_normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    epsilon: float = 1e-7,
    nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate (band_a - band_b) / (band_a + band_b) with zero-division safety and range clipping."""
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)

    numerator = a - b
    denominator = a + b

    # Avoid zero division and near-zero noise
    valid_denominator = np.abs(denominator) > epsilon
    result = np.zeros_like(numerator, dtype=np.float32)

    np.divide(numerator, denominator, out=result, where=valid_denominator)

    # Clip to physical index domain [-1.0, 1.0]
    np.clip(result, -1.0, 1.0, out=result)

    if nodata_mask is not None:
        result[~nodata_mask] = np.nan

    return result


def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    epsilon: float = 1e-7,
    nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI).
    
    Formula: (NIR - Red) / (NIR + Red)
    Dense vegetation typically yields NDVI > 0.4.
    Bare soil / built-up typically yields 0.05 < NDVI < 0.25.
    Water typically yields NDVI < 0.
    """
    return safe_normalized_difference(nir, red, epsilon=epsilon, nodata_mask=nodata_mask)


def calculate_ndbi(
    swir: np.ndarray,
    nir: np.ndarray,
    epsilon: float = 1e-7,
    nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate Normalized Difference Built-up Index (NDBI).
    
    Formula: (SWIR - NIR) / (SWIR + NIR)
    Urban / built-up impervious surfaces typically yield NDBI > 0.0.
    Vegetated land typically yields negative NDBI (< -0.1).
    """
    return safe_normalized_difference(swir, nir, epsilon=epsilon, nodata_mask=nodata_mask)


def calculate_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    epsilon: float = 1e-7,
    nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate Modified Normalized Difference Water Index (MNDWI).
    
    Formula: (Green - SWIR) / (Green + SWIR)
    Open water bodies typically yield MNDWI > 0.0.
    """
    return safe_normalized_difference(green, swir, epsilon=epsilon, nodata_mask=nodata_mask)


@dataclass
class SpectralBandCube:
    """Container for aligned multispectral reflectance bands."""
    blue: np.ndarray            # Sentinel-2 B02 (490 nm)
    green: np.ndarray           # Sentinel-2 B03 (560 nm)
    red: np.ndarray             # Sentinel-2 B04 (665 nm)
    nir: np.ndarray             # Sentinel-2 B08 (842 nm)
    swir: Optional[np.ndarray] = None  # Sentinel-2 B11 (1610 nm)
    nodata_mask: Optional[np.ndarray] = None
    acquisition_date: Optional[str] = None
    sensor: str = "Sentinel-2 MSI"

    def __post_init__(self):
        h, w = self.red.shape[:2]
        for name, band in [("blue", self.blue), ("green", self.green), ("nir", self.nir)]:
            if band.shape[:2] != (h, w):
                raise ValueError(f"Band dimension mismatch for {name}: expected {(h, w)}, got {band.shape[:2]}")
        if self.swir is not None and self.swir.shape[:2] != (h, w):
            raise ValueError(f"SWIR dimension mismatch: expected {(h, w)}, got {self.swir.shape[:2]}")

        # Ensure reflectance normalization in [0.0, 1.0] if provided as raw DN (e.g. 0-10000)
        for attr in ["blue", "green", "red", "nir", "swir"]:
            arr = getattr(self, attr)
            if arr is not None and arr.max() > 2.0:
                # Raw satellite reflectance digital numbers (0 - 10000 range)
                setattr(self, attr, np.clip(arr / 10000.0, 0.0, 1.0).astype(np.float32))
            elif arr is not None and arr.dtype != np.float32:
                setattr(self, attr, arr.astype(np.float32))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.red.shape[:2]

    def compute_ndvi(self) -> np.ndarray:
        return calculate_ndvi(self.nir, self.red, nodata_mask=self.nodata_mask)

    def compute_ndbi(self) -> Optional[np.ndarray]:
        if self.swir is None:
            return None
        return calculate_ndbi(self.swir, self.nir, nodata_mask=self.nodata_mask)

    def compute_mndwi(self) -> Optional[np.ndarray]:
        if self.swir is None:
            return None
        return calculate_mndwi(self.green, self.swir, nodata_mask=self.nodata_mask)


@dataclass
class MultispectralAttributionResult:
    """Analytical summary and spatial classification maps from multispectral attribution."""
    total_valid_pixels: int
    raw_rgb_change_pixels: int
    verified_sprawl_pixels: int
    agricultural_phenology_pixels: int
    water_dynamics_pixels: int
    bare_soil_excavation_pixels: int
    unverified_change_pixels: int

    # Pixel fraction metrics
    sprawl_attribution_ratio: float      # verified_sprawl / raw_rgb_change
    agricultural_false_alarm_ratio: float # phenology / raw_rgb_change

    # Detailed 2D discrete attribution classification map
    # Values: 0 = No change, 1 = Urban Sprawl, 2 = Agricultural Phenology, 3 = Water, 4 = Excavation
    attribution_mask: np.ndarray
    ndvi_t1: np.ndarray
    ndvi_t2: np.ndarray
    ndbi_t1: Optional[np.ndarray] = None
    ndbi_t2: Optional[np.ndarray] = None
    delta_ndvi: Optional[np.ndarray] = None
    delta_ndbi: Optional[np.ndarray] = None


def attribute_urban_sprawl(
    rgb_change_mask: np.ndarray,
    t1_cube: SpectralBandCube,
    t2_cube: SpectralBandCube,
    min_ndbi_post: float = -0.05,
    min_delta_ndbi: float = 0.08,
    max_ndvi_post: float = 0.30,
    crop_delta_ndvi_threshold: float = 0.20,
    nodata_mask: Optional[np.ndarray] = None
) -> MultispectralAttributionResult:
    """Attribute deep-learning binary change detections to specific land-conversion mechanisms.
    
    Decision Rules:
    -------------------------------------------------------------------------
    1. VERIFIED URBAN SPRAWL (Class 1):
       - Candidate pixel identified as changed by RGB model (rgb_change_mask == True)
       - Built-up signature increase: (NDBI_T2 - NDBI_T1) >= min_delta_ndbi OR NDBI_T2 >= min_ndbi_post
       - Loss of vegetation: NDVI_T2 <= max_ndvi_post
       - Concrete/asphalt impervious surface spectral signature is verified.
       
    2. AGRICULTURAL PHENOLOGY / CROP ROTATION (Class 2):
       - Candidate pixel identified as changed by RGB model (rgb_change_mask == True)
       - BUT built-up signature is absent: NDBI_T2 < min_ndbi_post AND delta_NDBI < min_delta_ndbi
       - Large seasonal vegetation swing: |NDVI_T2 - NDVI_T1| >= crop_delta_ndvi_threshold
       - Filters out false alarms from harvest, tilling, or crop cycles (e.g. Saclay, Valencia).
       
    3. WATER DYNAMICS (Class 3):
       - Candidate changed pixel where MNDWI_T2 > 0.1 or delta_MNDWI > 0.25.
       
    4. BARE SOIL / EXCAVATION (Class 4):
       - Candidate changed pixel with low vegetation (NDVI_T2 < 0.2) and intermediate NDBI (-0.15 to -0.05).
       - Signals pre-construction clearing or grading.
       
    5. UNVERIFIED SURFACE CHANGE (Class 5):
       - Changed in RGB, but SWIR band was not available or spectral criteria inconclusive.
    """
    h, w = rgb_change_mask.shape
    combined_valid = np.ones((h, w), dtype=bool)
    if nodata_mask is not None:
        combined_valid &= nodata_mask
    if t1_cube.nodata_mask is not None:
        combined_valid &= t1_cube.nodata_mask
    if t2_cube.nodata_mask is not None:
        combined_valid &= t2_cube.nodata_mask

    # Compute Spectral Indices
    ndvi1 = t1_cube.compute_ndvi()
    ndvi2 = t2_cube.compute_ndvi()
    delta_ndvi = ndvi2 - ndvi1

    has_swir = (t1_cube.swir is not None) and (t2_cube.swir is not None)
    if has_swir:
        ndbi1 = t1_cube.compute_ndbi()
        ndbi2 = t2_cube.compute_ndbi()
        delta_ndbi = ndbi2 - ndbi1
        mndwi1 = t1_cube.compute_mndwi()
        mndwi2 = t2_cube.compute_mndwi()
        delta_mndwi = mndwi2 - mndwi1
    else:
        ndbi1 = None
        ndbi2 = None
        delta_ndbi = None
        mndwi1 = None
        mndwi2 = None
        delta_mndwi = None

    # Classification raster: 0 = None, 1 = Sprawl, 2 = Agricultural, 3 = Water, 4 = Excavation, 5 = Unverified
    attribution_map = np.zeros((h, w), dtype=np.uint8)
    candidates = (rgb_change_mask > 0) & combined_valid

    if has_swir and ndbi1 is not None and ndbi2 is not None:
        # 1. Check for Water Dynamics
        water_mask = candidates & ((mndwi2 > 0.10) | (delta_mndwi > 0.25))
        attribution_map[water_mask] = 3

        # 2. Check for Verified Urban Sprawl (Built-up expansion)
        # Impervious surfaces require positive built-up index (NDBI_T2 >= min_ndbi_post)
        # AND a positive increase in built-up density (delta_ndbi >= min_delta_ndbi)
        sprawl_criteria = (
            candidates &
            ~water_mask &
            (ndbi2 >= min_ndbi_post) &
            (delta_ndbi >= min_delta_ndbi) &
            (ndvi2 <= max_ndvi_post)
        )
        attribution_map[sprawl_criteria] = 1

        # 3. Check for Agricultural Phenology (Crop harvest / vegetation change)
        agri_criteria = (
            candidates &
            ~water_mask &
            ~sprawl_criteria &
            ((np.abs(delta_ndvi) >= crop_delta_ndvi_threshold) | (ndvi1 > 0.35)) &
            (ndbi2 < min_ndbi_post)
        )
        attribution_map[agri_criteria] = 2

        # 4. Check for Bare Soil Excavation (Ground clearing / grading)
        soil_criteria = (
            candidates &
            ~water_mask &
            ~sprawl_criteria &
            ~agri_criteria &
            (ndvi2 < 0.22) &
            (ndbi2 >= -0.15) & (ndbi2 < min_ndbi_post)
        )
        attribution_map[soil_criteria] = 4

        # 5. Remaining candidates are unverified change
        unverified_criteria = candidates & (attribution_map == 0)
        attribution_map[unverified_criteria] = 5

    else:
        # SWIR is unavailable: Use NDVI difference heuristics as fallback
        sprawl_criteria = candidates & (ndvi2 < max_ndvi_post) & (delta_ndvi < -0.15)
        agri_criteria = candidates & ((delta_ndvi > 0.2) | (ndvi2 > 0.35))
        unverified = candidates & ~sprawl_criteria & ~agri_criteria

        attribution_map[sprawl_criteria] = 1
        attribution_map[agri_criteria] = 2
        attribution_map[unverified] = 5

    # Pixel Count Statistics
    raw_change = int(np.sum(candidates))
    sprawl_cnt = int(np.sum(attribution_map == 1))
    agri_cnt = int(np.sum(attribution_map == 2))
    water_cnt = int(np.sum(attribution_map == 3))
    soil_cnt = int(np.sum(attribution_map == 4))
    unverified_cnt = int(np.sum(attribution_map == 5))
    total_valid = int(np.sum(combined_valid))

    sprawl_ratio = round(sprawl_cnt / max(1, raw_change), 4)
    agri_ratio = round(agri_cnt / max(1, raw_change), 4)

    logger.info(
        f"Attribution complete: Raw change = {raw_change:,} px. "
        f"Verified Sprawl = {sprawl_cnt:,} ({sprawl_ratio*100:.1f}%), "
        f"Agricultural Phenology = {agri_cnt:,} ({agri_ratio*100:.1f}%), "
        f"Water = {water_cnt:,}, Soil = {soil_cnt:,}, Unverified = {unverified_cnt:,}."
    )

    return MultispectralAttributionResult(
        total_valid_pixels=total_valid,
        raw_rgb_change_pixels=raw_change,
        verified_sprawl_pixels=sprawl_cnt,
        agricultural_phenology_pixels=agri_cnt,
        water_dynamics_pixels=water_cnt,
        bare_soil_excavation_pixels=soil_cnt,
        unverified_change_pixels=unverified_cnt,
        sprawl_attribution_ratio=sprawl_ratio,
        agricultural_false_alarm_ratio=agri_ratio,
        attribution_mask=attribution_map,
        ndvi_t1=ndvi1,
        ndvi_t2=ndvi2,
        ndbi_t1=ndbi1,
        ndbi_t2=ndbi2,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi
    )


def generate_colorized_spectral_map(
    index_array: np.ndarray,
    color_scheme: str = "ndvi"
) -> np.ndarray:
    """Convert a 2D float index map in [-1.0, 1.0] to a standardized RGB visual raster [H, W, 3] uint8."""
    norm = np.clip((index_array + 1.0) / 2.0, 0.0, 1.0)  # map [-1, 1] to [0, 1]
    h, w = index_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if color_scheme == "ndvi":
        # NDVI: Brown/Red (low/no veg) -> Yellow (moderate) -> Dark Green (dense veg)
        for i in range(h):
            for j in range(w):
                v = norm[i, j]
                if np.isnan(v):
                    continue
                if v < 0.5:  # water / barren (-1.0 to 0.0)
                    t = v * 2.0
                    rgb[i, j] = [int(180 * (1 - t) + 210 * t), int(50 * (1 - t) + 180 * t), int(50 * (1 - t) + 140 * t)]
                else:        # vegetation (0.0 to 1.0)
                    t = (v - 0.5) * 2.0
                    rgb[i, j] = [int(210 * (1 - t) + 34 * t), int(180 * (1 - t) + 139 * t), int(140 * (1 - t) + 34 * t)]
    elif color_scheme == "ndbi":
        # NDBI: Green/Cyan (negative / veg) -> Gray/White (neutral) -> Bright Magenta/Red (built-up)
        for i in range(h):
            for j in range(w):
                v = norm[i, j]
                if np.isnan(v):
                    continue
                if v < 0.5:
                    t = v * 2.0
                    rgb[i, j] = [int(30 * (1 - t) + 160 * t), int(140 * (1 - t) + 160 * t), int(100 * (1 - t) + 160 * t)]
                else:
                    t = (v - 0.5) * 2.0
                    rgb[i, j] = [int(160 * (1 - t) + 235 * t), int(160 * (1 - t) + 50 * t), int(160 * (1 - t) + 70 * t)]
    else:
        # Grayscale fallback
        val_u8 = (norm * 255).astype(np.uint8)
        rgb[:, :, 0] = val_u8
        rgb[:, :, 1] = val_u8
        rgb[:, :, 2] = val_u8

    return rgb
