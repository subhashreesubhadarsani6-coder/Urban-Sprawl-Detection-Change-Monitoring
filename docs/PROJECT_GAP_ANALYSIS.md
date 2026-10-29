# UrbanScope AI — Project Gap Analysis & Technical Roadmap

**Audit Date**: October 29, 2026  
**Auditor**: Senior Full-Stack & Deep Learning Systems Architect  
**Project**: UrbanScope AI — Satellite Urban Sprawl Detection & Change Monitoring  

---

## 1. Executive Gap Assessment

UrbanScope AI possesses a fully verified core pipeline: a PyTorch FC-Siam-diff Siamese U-Net running with native CUDA 12.4 acceleration, an automated multispectral attribution engine (NDVI/NDBI), a FastAPI backend with persistent SQLite storage, and a React 18 GIS workstation with sub-second dynamic re-thresholding.

However, an objective audit reveals specific boundaries between **genuinely working functionality**, **mocked/simulated UI capabilities**, and **external infrastructure prerequisites**.

---

## 2. Comprehensive Feature Classification Matrix

| Feature / Subsystem | Current State | Verification Evidence | Operational Reality |
| :--- | :--- | :--- | :--- |
| **`FC-Siam-diff` PyTorch Model** | **Implemented & Verified** | `models/siam_unet.py`, `tests/test_model.py` (6 tests passing) | Real 7.76M parameter neural network; executes forward passes with exact shapes. |
| **CUDA AMP GPU Acceleration** | **Implemented & Verified** | PyTorch 2.6.0+cu124 on RTX 3050 Laptop GPU | 85× faster than CPU; peak VRAM 555 MiB; ~412 ms latency. |
| **Sliding-Window Inference Engine**| **Implemented & Verified** | `backend/app/services/geospatial_engine.py`, `tests/test_geospatial.py` | 2D Hann window blending eliminates seams; handles arbitrary dimensions. |
| **Resolution-Aware Area Engine** | **Implemented & Verified** | `geospatial_engine.py#L125-L160` | Extracts UTM projected meters and WGS84 geodesic degrees; flags uncalibrated rasters. |
| **Dynamic Threshold Re-evaluation**| **Implemented & Verified** | `POST /api/v1/analysis/{id}/rethreshold` | Sub-second recalculation without re-running neural forward passes. |
| **Multispectral Attribution (NDVI/NDBI)** | **Implemented & Verified** | `backend/app/geospatial/spectral_indices.py`, `tests/test_spectral_indices.py` | Eliminates agricultural harvest false alarms when 4+ bands are present. |
| **Multi-Format GIS Exports** | **Implemented & Verified** | Output files: `change_mask.tif`, `sprawl_boundaries.geojson`, `change_overlay.png`, CSV | Preserves CRS, geotransforms, and nodata tags. Verified valid by Rasterio. |
| **Interactive GIS Workstation** | **Implemented & Verified** | React 18 + Vite 5 + Tailwind in `src/`, production build in `dist/` | Swipe compare, threshold slider, layer toggles, and KPIs update dynamically. |
| **Browser Playwright E2E Tests** | **Implemented & Verified** | `tests/test_e2e_playwright.py` (2 tests passing) | Full 10-step user scenario verified against live running servers in 12.37s. |
| **Copernicus / Sentinel Hub Ingestion** | **Partially Implemented** | `backend/app/services/satellite_provider.py` | Provider abstraction and catalog search working; live downloads require external account quota. |
| **ChangeFormer & BIT-CD Architectures**| **UI Mock / Staged** | Dropdown options in `AnalysisUploadPage.tsx` and `backend/app/models/benchmark.py` | Models do not exist in source code; selecting them currently uses the pre-seeded benchmark values. |
| **User Authentication & RBAC** | **Missing (By Design)** | `backend/app/main.py` has no JWT/OAuth middleware | Safe for local GIS workstation; required if exposed publicly. |
| **Containerized Docker Runtime** | **Implemented but Not Verified** | `docker-compose.yml`, `Dockerfile.backend`, `Dockerfile.frontend` | Configured cleanly; unverified on host because Docker CLI is not installed on PATH. |

---

## 3. Prioritized Gap Analysis & Issue Log

| Priority | Component / Defect | Evidence & Location | Practical Impact | Recommended Action | Dependencies |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P1 (High)** | **Disclose Mock Models in UI** | `ChangeFormer` and `BIT-CD` appear in `AnalysisUploadPage.tsx#L293-L297` without source code implementations | Users might assume selecting ChangeFormer runs a Transformer model rather than FC-Siam-diff. | Add a badge `"(Reference Benchmark)"` in dropdown or disable selection until architectures are trained. | Frontend UI |
| **P2 (Medium)**| **Live Sentinel-2 Process API Quota** | `SentinelHubProvider._get_token()` in `satellite_provider.py#L90-L125` | If client credentials expire or exhaust free-tier limits, live acquisition fails. | Keep `MockSatelliteProvider` fallback active and provide clear instructions for user-supplied API keys. | External Sentinel Hub account |
| **P3 (Medium)**| **Docker Runtime Host Verification** | Docker CLI not present on local host PATH (`docker : The term 'docker' is not recognized`) | Containers cannot be tested locally without Docker Desktop. | Install Docker Desktop or run container verification on an external Linux staging server. | Docker Desktop / Linux host |
| **P4 (Low)** | **Persistent User Auth (RBAC)** | No login or session management in `backend/app/main.py` | Anyone on the local network can queue analysis jobs or re-threshold results. | Add FastAPI OAuth2 Password Bearer with JWT tokens if public deployment is required. | FastAPI security dependencies |
| **P5 (Low)** | **Multi-Spectral Native Model** | Siamese model only accepts 3-band RGB input; bands 8 & 11 are processed post-hoc via rules | Does not learn deep non-linear multispectral interaction features during backpropagation. | Train an 8-band or 12-band Siamese network when a multi-spectral change dataset is curated. | Compute & Dataset |

---

## 4. Practical Technical Roadmap

```mermaid
flowchart TD
    subgraph Phase 1: Operational Hygiene [Phase 1: Operational Hygiene (Immediate)]
        A1["Clarify UI Model Selection (Mark ChangeFormer as Benchmark)"]
        A2["Maintain Offline Mock Provider Fallback"]
    end

    subgraph Phase 2: Deployment & Infrastructure [Phase 2: Deployment & Infrastructure]
        B1["Install Docker Desktop / Verify Linux Container Parity"]
        B2["Add Optional JWT Authentication Layer for Multi-User Access"]
    end

    subgraph Phase 3: Scientific Scaling [Phase 3: Scientific Scaling]
        C1["Train Native 12-Band Multispectral Siamese Transformer"]
        C2["Connect Live Copernicus Data Space API Ingestion"]
        C3["Migrate SQLite to Managed PostGIS for Million-Polygon Vector Scale"]
    end

    Phase 1: Operational Hygiene --> Phase 2: Deployment & Infrastructure
    Phase 2: Deployment & Infrastructure --> Phase 3: Scientific Scaling
```

### Phase 1: Immediate Polish (Completed / Current)
- [x] All 59 unit, integration, geospatial, and browser E2E tests passing.
- [x] Purge temporary `.crdownload` fragments (1.17 GB freed).
- [x] Move hardcoded API secrets from code to `.env`.
- [x] Integrate multispectral NDVI/NDBI attribution to suppress agricultural harvest false alarms.

### Phase 2: Production Hardening
- [ ] Install Docker Desktop on Windows host or deploy to a remote Ubuntu GPU instance to test `docker compose up --build`.
- [ ] Add JWT authentication middleware in FastAPI (`backend/app/routers/auth.py`) for multi-tenant deployments.
- [ ] Restrict UI architecture dropdown to `FC-Siam-diff (Production)` while labeling others as comparative benchmarks.

### Phase 3: Research & Scientific Extensions
- [ ] Curate a native 12-band Sentinel-2 change detection dataset (incorporating B08 NIR and B11 SWIR directly into the encoder).
- [ ] Train a hierarchical Siamese Transformer (`ChangeFormer`) on multi-spectral imagery to improve subtle building edge boundary delineation.
- [ ] Switch database backend from SQLite to PostgreSQL + PostGIS for spatial SQL polygon intersection queries.
