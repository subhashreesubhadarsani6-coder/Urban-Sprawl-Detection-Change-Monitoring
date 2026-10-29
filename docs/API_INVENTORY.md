# UrbanScope AI — Comprehensive REST API Reference & Inventory

**Audit Date**: October 29, 2026  
**Auditor**: Senior Backend Architect & API Security Specialist  
**Base URL**: `http://127.0.0.1:8000/api/v1`  
**OpenAPI Specification**: Interactive Swagger UI at `http://127.0.0.1:8000/docs`  

---

## 1. API Architecture & Global Conventions

- **Protocol**: RESTful HTTP/1.1 JSON (UTF-8 encoded).
- **Authentication**: Currently configured for local GIS workstation operation (public access within CORS-whitelisted origins). Production deployments can enforce JWT Bearer tokens.
- **Validation**: Strict schema validation using Pydantic v2 models. Input validation errors return HTTP 422 with exact field locations.
- **Error Handling**: Standardized RFC 7807 problem details with `{"detail": "Error description"}` and HTTP status codes (`400`, `401`, `404`, `422`, `500`).
- **Path Traversal Protection**: All artifact and upload download endpoints explicitly sanitize filenames and enforce containment within designated storage roots using `Path.resolve()`.

---

## 2. Complete Endpoint Inventory (25 Endpoints)

### 2.1 System Health & Service Telemetry (`health.py`)

#### 1. `GET /api/v1/health`
- **Purpose**: Liveness and database connectivity health probe.
- **Source**: [`backend/app/routers/health.py#L12-L22`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/health.py#L12-L22) (`get_health`)
- **Calling Component**: Docker healthcheck, frontend status monitor, and test suites.
- **Test Coverage**: [`backend/tests/test_health.py::test_health_check`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_health.py) (PASSED).
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "database": "connected",
    "timestamp": "2026-10-29T10:11:37.211592+00:00"
  }
  ```

#### 2. `GET /api/v1/status`
- **Purpose**: System resource monitoring (GPU model, VRAM usage, active jobs, storage).
- **Source**: [`backend/app/routers/health.py#L24-L52`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/health.py#L24-L52) (`get_system_status`)
- **Calling Component**: [`src/components/layout/Header.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/components/layout/Header.tsx)
- **Test Coverage**: [`backend/tests/test_health.py::test_system_status`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_health.py) (PASSED).
- **Response Example**:
  ```json
  {
    "engine_status": "Ready",
    "active_gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "gpu_memory_used_mb": 555,
    "gpu_memory_total_mb": 4095,
    "active_jobs_count": 0,
    "sentinel_api_connected": true,
    "storage_used_mb": 42.8
  }
  ```

---

### 2.2 Areas of Interest (AOI) Management (`aoi.py`)

#### 3. `GET /api/v1/aoi`
- **Purpose**: Retrieve all registered metropolitan monitoring regions.
- **Source**: [`backend/app/routers/aoi.py#L11-L15`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/aoi.py#L11-L15) (`get_all_aois`)
- **Calling Component**: [`src/pages/DashboardPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/DashboardPage.tsx), [`src/pages/AnalysisUploadPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/AnalysisUploadPage.tsx)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_get_aois`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).
- **Response Example**:
  ```json
  [
    {
      "id": "aoi-las-vegas",
      "name": "Las Vegas Valley Southwest Expansion",
      "city": "Las Vegas",
      "country": "United States",
      "centerLat": 36.1699,
      "centerLng": -115.1398,
      "zoom": 12,
      "bbox": {"minLat": 36.05, "minLng": -115.35, "maxLat": 36.25, "maxLng": -115.10},
      "monitoredSince": "2018-01-01",
      "totalAreaKm2": 712.4,
      "urbanGrowthRateAnnual": 4.8,
      "riskLevel": "High",
      "thumbnailUrl": "/thumbnails/las-vegas.png"
    }
  ]
  ```

#### 4. `GET /api/v1/aoi/{aoi_id}`
- **Purpose**: Fetch detailed geographic metadata for a single AOI.
- **Source**: [`backend/app/routers/aoi.py#L17-L22`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/aoi.py#L17-L22) (`get_aoi_by_id`)
- **Test Coverage**: Covered in integration tests.

#### 5. `POST /api/v1/aoi`
- **Purpose**: Register a new custom Area of Interest.
- **Source**: [`backend/app/routers/aoi.py#L24-L35`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/aoi.py#L24-L35) (`create_aoi`)
- **Status Code**: `201 Created`

---

### 2.3 Analysis & Change Detection Pipeline (`analysis.py`)

#### 6. `GET /api/v1/analysis`
- **Purpose**: List analysis jobs with optional status filtering (`?status=completed`).
- **Source**: [`backend/app/routers/analysis.py#L18-L26`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L18-L26) (`get_analysis_jobs`)
- **Calling Component**: [`src/pages/DashboardPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/DashboardPage.tsx)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_get_analysis_jobs`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).

#### 7. `GET /api/v1/analysis/{job_id}`
- **Purpose**: Retrieve full analytical results, metrics, and artifact links for a job.
- **Source**: [`backend/app/routers/analysis.py#L28-L34`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L28-L34) (`get_analysis_job_by_id`)
- **Calling Component**: [`src/pages/ResultsPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/ResultsPage.tsx)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_get_single_job_details`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).

#### 8. `GET /api/v1/analysis/{job_id}/progress`
- **Purpose**: Lightweight polling endpoint for progress bar and pipeline step status.
- **Source**: [`backend/app/routers/analysis.py#L36-L45`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L36-L45) (`get_job_progress`)
- **Calling Component**: Results page auto-refresher (polls every 1.5s during execution).

#### 9. `POST /api/v1/analysis`
- **Purpose**: Queue a bi-temporal change detection run; launches background worker.
- **Source**: [`backend/app/routers/analysis.py#L65-L95`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L65-L95) (`create_analysis_job`)
- **Calling Component**: [`src/pages/AnalysisUploadPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/AnalysisUploadPage.tsx)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_create_and_poll_analysis_job`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).
- **Status Code**: `202 Accepted`
- **Request Payload Example**:
  ```json
  {
    "title": "Las Vegas Southwest Expansion Audit",
    "description": "Tracking suburban fringe conversion",
    "aoiId": "aoi-las-vegas",
    "parameters": {
      "modelArch": "FC-Siam-diff",
      "confidenceThreshold": 0.50,
      "calculateNdvi": true,
      "calculateNdbi": true
    },
    "imagery": {
      "preDate": "2018-06-15",
      "postDate": "2024-06-20",
      "sensor": "Sentinel-2 MSI",
      "resolutionMeters": 10.0,
      "preImageUrl": "/storage/uploads/t1_lasvegas.tif",
      "postImageUrl": "/storage/uploads/t2_lasvegas.tif"
    }
  }
  ```

#### 10. `POST /api/v1/analysis/{job_id}/rethreshold?threshold=0.70`
- **Purpose**: Sub-second re-evaluation of binary mask and $\text{km}^2$ area from cached continuous probability map without re-running neural network inference.
- **Source**: [`backend/app/routers/analysis.py#L97-L135`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L97-L135) (`rethreshold_analysis_job`)
- **Calling Component**: [`src/pages/ResultsPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/ResultsPage.tsx) (Dynamic Threshold Slider)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_rethreshold_job_endpoint`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).

#### 11. `GET /api/v1/analysis/{job_id}/export/csv`
- **Purpose**: Download comprehensive tabular metrics and LULC transition table as CSV.
- **Source**: [`backend/app/routers/analysis.py#L47-L63`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/analysis.py#L47-L63) (`export_job_metrics_csv`)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_export_csv_endpoint`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).

---

### 2.4 File Uploads & Artifact Downloads (`uploads.py`)

#### 12. `POST /api/v1/upload`
- **Purpose**: Upload satellite raster files (`.tif`, `.tiff`, `.png`, `.jpg`, `.jp2`).
- **Source**: [`backend/app/routers/uploads.py#L15-L38`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/uploads.py#L15-L38) (`upload_satellite_image`)
- **Constraints**: Size limited to 50 MB; extensions restricted; UUID prefix prepended to filename to avoid collisions.
- **Test Coverage**: [`backend/tests/test_uploads.py`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_uploads.py) (3 tests passing).

#### 13. `GET /api/v1/storage/uploads/{filename}`
- **Purpose**: Retrieve an uploaded raw satellite raster file with path traversal sanitization.
- **Source**: [`backend/app/routers/uploads.py#L40-L48`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/uploads.py#L40-L48) (`get_uploaded_file`)

#### 14. `GET /api/v1/artifacts/{job_id}/{filename}`
- **Purpose**: Download generated analysis artifacts (`change_mask.tif`, `sprawl_boundaries.geojson`, `change_overlay.png`, `probability_map.png`, `audit_report.txt`, `ndvi_t1.png`).
- **Source**: [`backend/app/routers/uploads.py#L50-L60`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/uploads.py#L50-L60) (`get_job_artifact`)
- **Test Coverage**: [`backend/tests/test_analysis.py::test_download_artifacts_and_path_traversal_protection`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_analysis.py) (PASSED).

---

### 2.5 Satellite Acquisition & Ingestion (`satellite.py`)

#### 15. `GET /api/v1/satellite/provider/health`
- **Purpose**: Check Copernicus Data Space / Sentinel Hub OAuth2 API connectivity and token status.
- **Source**: [`backend/app/routers/satellite.py#L43-L52`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/satellite.py#L43-L52) (`get_provider_health`)
- **Calling Component**: [`src/pages/AnalysisUploadPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/AnalysisUploadPage.tsx)
- **Test Coverage**: [`backend/tests/test_satellite.py::test_satellite_api_endpoints`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/tests/test_satellite.py) (PASSED).

#### 16. `POST /api/v1/satellite/search`
- **Purpose**: Search cloud-filtered Sentinel-2 scenes across bounding box coordinates and dates.
- **Source**: [`backend/app/routers/satellite.py#L54-L77`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/satellite.py#L54-L77) (`search_satellite_scenes`)
- **Request Payload**:
  ```json
  {
    "min_lon": -115.35, "min_lat": 36.05,
    "max_lon": -115.10, "max_lat": 36.25,
    "date_from": "2018-06-01", "date_to": "2024-06-01",
    "max_cloud_cover": 10.0
  }
  ```

#### 17. `POST /api/v1/satellite/acquire`
- **Purpose**: Stage and download a bi-temporal scene pair for direct deep learning change detection.
- **Source**: [`backend/app/routers/satellite.py#L79-L113`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/satellite.py#L79-L113) (`acquire_satellite_pair`)

---

### 2.6 Datasets, Training & Evaluation Catalogs

#### 18. `GET /api/v1/datasets`
- **Purpose**: Retrieve satellite dataset catalog items with sensor and keyword filtering.
- **Source**: [`backend/app/routers/datasets.py#L11-L22`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/datasets.py#L11-L22)
- **Calling Component**: [`src/pages/DatasetExplorerPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/DatasetExplorerPage.tsx)

#### 19. `GET /api/v1/datasets/{dataset_id}`
- **Purpose**: Retrieve single dataset metadata.

#### 20. `GET /api/v1/training/active` & `GET /api/v1/training`
- **Purpose**: Fetch active training telemetry, training loss curves, and validation metrics.
- **Source**: [`backend/app/routers/training.py#L11-L16`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/training.py#L11-L16)
- **Calling Component**: [`src/pages/ModelTrainingPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/ModelTrainingPage.tsx)

#### 21. `POST /api/v1/training/start`
- **Purpose**: Initiate a new neural network training run with custom hyperparameters.
- **Source**: [`backend/app/routers/training.py#L18-L38`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/training.py#L18-L38)

#### 22. `POST /api/v1/training/pause`
- **Purpose**: Toggle pause/resume on active training session.

#### 23. `GET /api/v1/evaluation/benchmarks` & `GET /api/v1/evaluation`
- **Purpose**: Retrieve cross-architecture benchmarking metrics (ChangeFormer, BIT-CD, FC-Siam-diff).
- **Source**: [`backend/app/routers/evaluation.py#L10-L15`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/evaluation.py#L10-L15)
- **Calling Component**: [`src/pages/EvaluationPage.tsx`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/pages/EvaluationPage.tsx)

#### 24. `GET /api/v1/evaluation/confusion-matrix`
- **Purpose**: Retrieve confusion matrix entries for validation visualization.
- **Source**: [`backend/app/routers/evaluation.py#L17-L22`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/routers/evaluation.py#L17-L22)

#### 25. `GET /`
- **Purpose**: Service greeting and navigation links to `/docs` and `/api/v1/health`.
- **Source**: [`backend/app/main.py#L74-L82`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/backend/app/main.py#L74-L82)
