# UrbanScope AI — Dataset & Remote Sensing Data Inventory

**Audit Date**: October 29, 2026  
**Auditor**: Senior Remote Sensing & GIS Data Engineer  
**Dataset Directory**: `C:\Project\Urban Sprawl Detection & Change Monitoring\dataset`  

---

## 1. Primary Dataset Overview

| Attribute | Specification & Evidence |
| :--- | :--- |
| **Dataset Name** | **Onera Satellite Change Detection (OSCD)** |
| **Source & Reference** | Daudt et al., *"Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks"*, IGARSS / IEEE GRSL, 2018. |
| **Origin & Platform** | European Space Agency (ESA) Copernicus **Sentinel-2 MSI** (Level-1C / Level-2A surface reflectance). |
| **Dataset Status** | Public Scientific Benchmark (Open Access for academic and non-commercial research). |
| **Target Application** | Urban sprawl detection, construction monitoring, and bi-temporal surface change segmentation. |
| **Local Storage Directory**| `dataset/` |
| **File Formats Present** | Apache Parquet (`.parquet`), Compressed Zip Archives (`.zip`), GeoTIFF (`.tif`), PNG (`.png`). |
| **Total Disk Footprint** | ~2.78 GB (Compressed Parquet shards and ground truth masks). |

---

## 2. Inventory of Dataset Files on Disk

```
dataset/
├── Onera Satellite Change Detection dataset - Test Labels.zip    (83.6 KB, Ground truth binary masks for test scenes)
├── Onera Satellite Change Detection dataset - Train Labels.zip   (137.9 KB, Ground truth binary masks for train scenes)
├── test-00000-of-00001.parquet                                 (11.56 MB, Self-contained test split archive)
├── test-00000-of-00003.parquet                                 (418.83 MB, High-res multi-shard test imagery Part 1)
├── test-00001-of-00003.parquet                                 (419.01 MB, High-res multi-shard test imagery Part 2)
├── test-00002-of-00003.parquet                                 (417.32 MB, High-res multi-shard test imagery Part 3)
├── train-00000-of-00001.parquet                                (24.70 MB, Self-contained train split archive)
├── train-00000-of-00005.parquet                                (508.23 MB, Multi-shard train imagery Part 1)
├── train-00001-of-00005.parquet                                (432.29 MB, Multi-shard train imagery Part 2)
├── train-00002-of-00005.parquet                                (473.78 MB, Multi-shard train imagery Part 3)
└── train-00004-of-00005.parquet                                (467.36 MB, Multi-shard train imagery Part 5)
```

> [!NOTE]
> **Data Integrity Observation**: `train-00003-of-00005.parquet` is absent from disk due to an aborted upstream download. The training and evaluation pipeline (`src/data/dataset.py`, `training/train.py`, `scripts/prepare_dataset.py`) intentionally reads from `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`, which are complete, self-contained single-shard representations of all 24 registered scenes.

---

## 3. Geographic Partitioning & Leakage Prevention

To ensure strict scientific validity and prevent spatial autocorrelation leakage, the 24 registered global scenes are strictly divided geographically:

```
Total Registered Scenes: 24 International Urban Corridors
├── Training Split (11 Scenes, 45.8%):
│   ├── abudhabi (UAE)        ├── aguasclaras (Brazil)   ├── beihai (China)
│   ├── bordeaux (France)     ├── cupertino (USA)        ├── mumbai (India)
│   ├── nantes (France)       ├── paris (France)         ├── pisa (Italy)
│   ├── rennes (France)       └── saclay_e (France)
│
├── Validation Split (3 Scenes, 12.5%):
│   ├── beirut (Lebanon)      ├── bercy (France)         └── hongkong (Hong Kong)
│
└── Held-Out Test Split (10 Scenes, 41.7%):
    ├── brasilia (Brazil)     ├── chongqing (China)      ├── dubai (UAE)
    ├── lasvegas (USA)        ├── milano (Italy)         ├── montpellier (France)
    ├── norcia (Italy)        ├── rio (Brazil)           ├── saclay_w (France)
    └── valencia (Spain)
```

- **Manifest Configuration**: Defined in [`configs/geographic_splits.json`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/configs/geographic_splits.json) and enforced in [`src/data/splits.py`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/data/splits.py).
- **Leakage Verification**: Automated assertions in [`tests/test_dataset.py`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/tests/test_dataset.py#L15-L35) verify that $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.
- **Validation Tuning Policy**: Probability threshold $\tau = 0.50$ was selected strictly on the 3 validation cities (`beirut`, `bercy`, `hongkong`). Held-out test scenes were never utilized for model tuning.

---

## 4. Ground-Truth Mask Structure & Class Imbalance

- **Target Label**: Single-channel 2D spatial binary mask (`uint8`).
  - `0`: Unchanged land surface (natural vegetation, permanent water, undisturbed desert, established urban buildings).
  - `1`: Changed land surface (new residential construction, commercial expansion, infrastructure paving, earthmoving).
  - `255 / NaN`: Nodata boundary padding (excluded from loss backpropagation and metric accumulation).
- **Class Imbalance Distribution**:
  - Valid Evaluated Pixels across 10 Test Cities: **3,077,936 px**
  - Actual Change Pixels ($\text{Class } 1$): **159,077 px** ($\approx 5.17\%$)
  - Unchanged Pixels ($\text{Class } 0$): **2,918,859 px** ($\approx 94.83\%$)
  - Negative-to-Positive Imbalance Ratio: **$\approx 18.3 : 1$**
- **Imbalance Mitigation**: Addressed during training via positive class weighting ($w_{\text{pos}} = 3.0$ in BCE loss) and Soft Dice loss ($w_{\text{dice}} = 0.5$).

---

## 5. End-to-End Data Pipeline Architecture

```
+---------------------------------------------------------------------------------------+
|                                1. Raw Dataset on Disk                                 |
|  Parquet Files (train-00000-of-00001.parquet, test-00000-of-00001.parquet)             |
|  Contains scene metadata, raster byte arrays (T1, T2), and compressed binary masks    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                            2. Extraction & Ingestion Engine                           |
|  src/data/dataset.py: ChangeDetectionDataset                                          |
|  - Reads parquet rows into memory-mapped NumPy arrays                                 |
|  - Filters scenes based on configs/geographic_splits.json                             |
|  - Masks out nodata borders (where all channels == 0)                                 |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                            3. Systematic Patching & Slicing                           |
|  - Tiling Strategy: 256 x 256 pixel square patches                                    |
|  - Training Stride: 160 px (dense sampling) or 224 px (standard)                      |
|  - Minimum Valid Ratio: Patches with < 50% valid pixels are rejected                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                       4. Synchronized Data Augmentation Pipeline                      |
|  src/data/transforms.py: SynchronizedTransforms                                       |
|  - Identical geometric transformation applied simultaneously to T1, T2, and Mask:     |
|    * Random Horizontal Flip (p = 0.5)                                                 |
|    * Random Vertical Flip (p = 0.5)                                                   |
|    * Random Orthogonal Rotations (90°, 180°, 270°, p = 0.5)                           |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                           5. Normalization & Tensor Packing                           |
|  - Scale raw digital numbers / byte values to [0.0, 1.0]                              |
|  - Standardize channels using ImageNet statistics:                                    |
|    Mean: [0.485, 0.456, 0.406]  |  Std: [0.229, 0.224, 0.225]                         |
|  - Format output as PyTorch tensors:                                                  |
|    t1: [4, 3, 256, 256], t2: [4, 3, 256, 256], mask: [4, 1, 256, 256]                |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                6. PyTorch Siamese Model                               |
|  models/siam_unet.py: FC-Siam-diff                                                    |
|  Forward pass: model(t1, t2) -> Logits [4, 1, 256, 256]                              |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                           7. Evaluation & Metric Accumulation                         |
|  src/metrics.py: ConfusionMatrixTracker                                               |
|  Accumulates TP, FP, TN, FN globally across all patches to compute F1, IoU, Precision |
+---------------------------------------------------------------------------------------+
```

---

## 6. How the Live Application Loads Data

- **Training & Benchmarks**: Directly queries the Parquet shards via `ChangeDetectionDataset`.
- **Live User Analysis (`/analysis/new`)**:
  - The user uploads arbitrary multi-temporal satellite files (PNG, JPEG, GeoTIFF) or selects a pre-configured Area of Interest (AOI).
  - If no custom files are provided, the system loads [`reports/sample_t1_lasvegas.png`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/reports/sample_t1_lasvegas.png) and [`reports/sample_t2_lasvegas.png`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/reports/sample_t2_lasvegas.png) ($716 \times 824 \text{ px}$ extracted from the OSCD Las Vegas scene).
  - The Geospatial Engine derives Ground Sample Distance (GSD), applies sliding-window tiled inference, and vectorizes change boundaries to GeoJSON.
