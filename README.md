# UrbanScope AI — Urban Sprawl Detection & Change Monitoring

> **Production-Grade Deep Learning & Geospatial Remote Sensing System**  
> Tracks, segments, and quantifies bi-temporal land-cover changes, impervious surface growth, and urban sprawl using a verified **Siamese U-Net (`FC-Siam-diff`)** neural network with full georeferencing preservation.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |            React 18 + Vite GIS Web UI       |
                                      | (Swipe Comparer, Opacity Controls, Charts)  |
                                      +----------------------+----------------------+
                                                             | HTTP / REST (port 5173 / 80)
                                                             v
+-----------------------------------------------------------------------------------+
|                              FastAPI REST Backend (port 8000)                     |
|                                                                                   |
|  +------------------------+  +------------------------+  +---------------------+  |
|  |     Job Ingestion      |  |   Geospatial Engine    |  |  Artifact Exporter  |  |
|  | (Size/Path Validation) |  | (CRS, GSD, Blending)   |  | (GeoTIFF, GeoJSON,  |  |
|  |                        |  |                        |  |  CSV, PNG, Text)    |  |
|  +------------------------+  +-----------+------------+  +---------------------+  |
+------------------------------------------|----------------------------------------+
                                           v
+-----------------------------------------------------------------------------------+
|                       PyTorch Siamese U-Net (FC-Siam-diff)                        |
|                                                                                   |
|  Earlier (T1) Image ---> [Shared Conv Encoder]                                    |
|                                \ (Feature Diff)                                   |
|  Later (T2) Image   ---> [Shared Conv Encoder] ---> [Decoder] ---> Change Logits  |
|                                                                                   |
|  Tiling Strategy: 256x256 tiles, 64px overlap, 2D Hanning window seamless blending |
|  Checkpoint: checkpoints/best_checkpoint.pth (7.76M params, Val F1 = 0.4871)     |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

1. **Siamese U-Net Change Detection**:
   - `FC-Siam-diff` architecture with weight-sharing dual encoder branches and multi-scale difference skip connections.
   - Evaluated across **3,077,936 test pixels** spanning 10 held-out international test cities (Montpellier, Las Vegas, Milano, Rio, etc.).
2. **Geospatial & Remote Sensing Integrity**:
   - Supports **GeoTIFF, TIFF (8/16-bit), PNG, and JPEG**.
   - Preserves projected Coordinate Reference Systems (e.g. UTM Zones `EPSG:32611`) and geotransform metadata.
   - Handles geographic coordinate systems (WGS84 `EPSG:4326`) with latitude-corrected geodesic calculations.
   - Computes calibrated urban change area in $\text{km}^2$; never assumes every image has 10m resolution.
3. **Interactive Visual GIS Workspace**:
   - Split-screen before/after swipe comparison tool.
   - Real-time probability threshold slider ($0.10$ to $0.90$) with sub-150ms re-thresholding.
   - Red change overlay and continuous probability heatmap toggles with opacity control ($10\%$–$100\%$).
   - Comprehensive provenance labeling (`Real Model Inference (FC-Siam-diff)` vs `Demo Output`).
4. **Multi-Format Export Pipeline**:
   - **GeoTIFF**: Binary change mask preserving CRS and spatial transform (`change_mask.tif`).
   - **GeoJSON**: Sprawl boundary vector polygons (`sprawl_boundaries.geojson`).
   - **CSV**: Tabular change metrics, LULC transition matrices, and temporal trend tables.
   - **PNG**: Transparent RGBA overlay (`change_overlay.png`) and probability heatmap (`probability_map.png`).
   - **Text / PDF**: Detailed analytical audit report (`audit_report.txt`).

---

## System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Windows 11 / Ubuntu 22.04 LTS |
| **Python** | Python 3.10 to 3.13 | Python 3.11 or 3.12 |
| **Node.js** | Node.js 18 LTS | Node.js 20+ LTS with npm 10+ |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Optional (CPU supported) | NVIDIA RTX (4GB+ VRAM) with CUDA 11.8 / 12.x |
| **Disk Space** | 5 GB available space | 15 GB (for datasets & checkpoints) |

---

## Quickstart: Windows Setup Instructions

### 1. Clone & Set Up Python Virtual Environment
Open PowerShell or CMD in the project root:

```powershell
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# In PowerShell:
.\.venv\Scripts\Activate.ps1
# In CMD:
.\.venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

> **Note on Windows GDAL/Rasterio**: Pre-compiled binary wheels for `rasterio` and `affine` install automatically via `pip install -r requirements.txt`. Ensure the Microsoft Visual C++ 2015–2022 Redistributable is installed on your machine.

### 2. Set Up Frontend Node Environment

```powershell
# Install Node.js frontend dependencies
npm install

# Build production assets (optional check)
npm run build
```

### 3. Start Backend & Frontend Development Servers

Open two separate terminal windows:

**Terminal 1 — FastAPI Backend:**
```powershell
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation is available at: `http://127.0.0.1:8000/docs`

**Terminal 2 — React GIS Frontend:**
```powershell
npm run dev
```
Open your browser at: `http://127.0.0.1:5173/`

---

## Quickstart: Linux / macOS Setup Instructions

```bash
# 1. Clone repository and navigate
cd "Urban Sprawl Detection & Change Monitoring"

# 2. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install frontend dependencies
npm install

# 4. Launch Backend (Terminal 1)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Launch Frontend (Terminal 2)
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## Deployment: Docker Compose

To deploy both frontend and backend as containerized microservices:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Build and start containers
docker compose up --build -d

# 3. View running container status
docker compose ps

# 4. View logs
docker compose logs -f
```

- **Frontend Application**: `http://localhost:5173` (or `http://localhost:80`)
- **Backend API & Swagger Docs**: `http://localhost:8000/docs`
- **Health Endpoint**: `http://localhost:8000/api/v1/health`

To stop the containers:
```bash
docker compose down
```

### 2. GPU Acceleration Setup (NVIDIA RTX with CUDA 12.4)
For Windows systems equipped with NVIDIA GPUs (such as RTX 3050 Laptop GPU / Desktop RTX series):

```powershell
# Install PyTorch with native CUDA 12.4 support (enables Tensor Core acceleration and AMP)
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA activation and GPU memory
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

> **Performance Note**: On NVIDIA RTX 3050 with Automatic Mixed Precision (AMP), training throughput increases by **~85x** (epoch duration reduces from 35s on CPU to ~0.4s on GPU).

---

## Model Training & Inference CLI

### 1. Dataset Preparation & Geographic Isolation Audit
Inspect parquet shards, verify class distributions, and generate strict geographic splits:

```bash
python scripts/inspect_environment_and_dataset.py
python scripts/prepare_dataset.py
```

### 2. Train Siamese U-Net (`FC-Siam-diff`)
Train modular FC-Siam-diff with BCE + Soft Dice loss, AMP acceleration, and early stopping:

```bash
python training/train.py --config training/config.yaml
```

To resume training from the latest recovery checkpoint:
```bash
python training/train.py --config training/config.yaml --resume
```

### 3. Run Controlled Optimization Experiments
Execute isolated parameter comparison experiments strictly over training and validation splits:

```bash
python scripts/run_experiments.py
```

### 4. Held-Out Scene Evaluation & Diagnostic Audit
Evaluate the trained checkpoint across all 10 held-out geographic test scenes and produce 8-panel error diagnostic maps:

```bash
python training/evaluate_test.py --checkpoint artifacts/checkpoints/best_model.pth
python scripts/audit_predictions.py
```

### 5. Command-Line Scene Prediction
Run seamless tiled inference with Hann window blending on arbitrary bi-temporal satellite image pairs:

```bash
python inference/predict.py \
  --t1 reports/sample_t1_lasvegas.png \
  --t2 reports/sample_t2_lasvegas.png \
  --checkpoint artifacts/checkpoints/best_model.pth \
  --threshold 0.50 \
  --output-dir reports/predictions
```

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status and database connectivity. |
| `GET` | `/api/v1/aoi` | List all monitored Areas of Interest. |
| `GET` | `/api/v1/analysis` | Retrieve all analysis jobs with status filtering. |
| `GET` | `/api/v1/analysis/{id}` | Full analysis details, metrics, and LULC matrix. |
| `GET` | `/api/v1/analysis/{id}/progress` | Lightweight polling endpoint for execution progress. |
| `POST`| `/api/v1/analysis` | Queue a new bi-temporal change detection job. |
| `POST`| `/api/v1/analysis/{id}/rethreshold` | Dynamically re-evaluate change mask at a new cutoff threshold. |
| `GET` | `/api/v1/analysis/{id}/export/csv` | Download full analysis metrics and transition tables as CSV. |
| `POST`| `/api/v1/upload` | Upload satellite raster or vector files with path sanitization. |
| `GET` | `/api/v1/artifacts/{id}/{filename}`| Download generated artifacts (GeoTIFF, GeoJSON, PNG, Text). |
| `GET` | `/api/v1/datasets` | Multi-sensor satellite dataset catalog. |
| `GET` | `/api/v1/evaluation/benchmarks`| Cross-architecture performance benchmarks. |
| `GET` | `/api/v1/satellite/provider/health` | Query Sentinel Hub / Copernicus provider connectivity and authentication status. |
| `POST`| `/api/v1/satellite/search` | Search cloud-filtered satellite scenes across AOI bounding box and dates. |
| `POST`| `/api/v1/satellite/acquire` | Stage and download paired bi-temporal satellite scenes for analysis. |

---

## Multispectral Urban-Sprawl Attribution (NDVI & NDBI)

While standard 3-band RGB Siamese networks identify visible surface differences, agricultural harvesting and seasonal crop cycles can trigger false-positive change detections. UrbanScope AI integrates an automated multispectral attribution engine:

1. **Normalized Difference Vegetation Index (NDVI)**:
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} \quad (\text{Sentinel-2 Band 8 and Band 4})$$
2. **Normalized Difference Built-up Index (NDBI)**:
   $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}} \quad (\text{Sentinel-2 Band 11 and Band 8})$$
3. **Decision Rules for True Sprawl vs Crop Phenology**:
   - **Verified Urban Sprawl**: Candidate change pixel exhibiting vegetation reduction ($\text{NDVI}_{T2} \le 0.30$) combined with structural built-up material ($\text{NDBI}_{T2} \ge -0.05$ and $\Delta\text{NDBI} \ge 0.08$).
   - **Agricultural Phenology**: Candidate pixel exhibiting vegetation fluctuation ($|\Delta\text{NDVI}| \ge 0.20$), but negative built-up index ($\text{NDBI}_{T2} < -0.05$), confirming the absence of concrete, asphalt, or masonry.

---

## Satellite Imagery Ingestion (Copernicus & Sentinel Hub)

UrbanScope AI supports both direct file uploads and automated satellite ingestion via the official Sentinel Hub / Copernicus Data Space API:
1. Configure credentials in `.env` (never check secrets into source control):
   ```ini
   SENTINEL_CLIENT_ID="sh-your-client-id"
   SENTINEL_CLIENT_SECRET="your-client-secret"
   SENTINEL_TOKEN_URL="https://services.sentinel-hub.com/oauth/token"
   ```
2. When credentials are not configured, the application automatically falls back to the `MockSatelliteProvider`, enabling full offline workflow testing and verification.

---

## Docker Deployment

### 1. CPU-Only Deployment
```bash
docker compose up --build
```
The application will launch backend services on port 8000 and the React production bundle via Nginx on port 80 (and 5173).

### 2. GPU-Enabled Deployment (Linux with NVIDIA Container Toolkit)
Ensure `nvidia-container-toolkit` is installed on the host. In `docker-compose.yml`, uncomment the `deploy` reservations:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## Automated Test Suite

UrbanScope AI includes an exhaustive 59-test automated test suite:

```powershell
pytest tests/ backend/tests/ -v
```

### Verified Test Breakdown:
- **Model Architecture & Tensors** (`tests/test_model.py`): 6 tests.
- **Loss Functions & Boundary Safety** (`tests/test_losses.py`): 4 tests.
- **Metrics & Exact Confusion Counts** (`tests/test_metrics.py`): 4 tests.
- **Geographic Scene Isolation** (`tests/test_dataset.py`): 3 tests.
- **Model Checkpointing & Invariance** (`tests/test_checkpoint.py`): 3 tests.
- **Geospatial Engine & Real Inference** (`tests/test_geospatial.py`): 9 tests (includes multispectral GeoTIFF attribution).
- **Spectral Indices & Sprawl Attribution** (`tests/test_spectral_indices.py`): 6 tests (NDVI/NDBI/MNDWI safe division and phenology separation).
- **FastAPI Endpoints, Uploads & Security** (`backend/tests/`): 22 tests (includes satellite provider search & health).
- **Browser Playwright E2E Automation** (`tests/test_e2e_playwright.py`): 2 tests (10-step full workflow + security tests).

**Total: 59 / 59 PASSED (100% Green)**

---

## Troubleshooting Guide

### 1. Rasterio / GDAL Import Errors on Windows
If `import rasterio` reports `DLL load failed`:
- Verify you are inside the virtual environment: `.\.venv\Scripts\python -c "import rasterio"`.
- Ensure the **Microsoft Visual C++ Redistributable (2015-2022)** is installed.
- Reinstall rasterio with pre-compiled wheels: `pip install --force-reinstall rasterio tifffile`.

### 2. PyTorch CUDA vs CPU Execution
- By default, PyTorch runs in CPU mode if CUDA is not detected.
- To enable NVIDIA GPU acceleration on Windows:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- The `ChangePredictor` automatically detects `torch.cuda.is_available()` and offloads tiled inference to GPU VRAM.

### 3. SQLite Database Concurrency
- SQLite is used for lightweight local persistence.
- If you encounter `database is locked`, ensure only one uvicorn worker process is writing (`--workers 1` when using SQLite).
- For high-concurrency production deployments, set `DATABASE_URL=postgresql://user:pass@host/dbname` in `.env`.

### 4. CORS Errors on Web Frontend
- If browser displays `Cross-Origin Request Blocked`:
  - Ensure backend is running on `http://127.0.0.1:8000`.
  - Check that `.env` contains `VITE_API_BASE_URL=http://localhost:8000`.
  - Backend CORS settings in `backend/app/config.py` allow `http://localhost:5173` and `http://127.0.0.1:5173`.
