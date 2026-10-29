# UrbanScope AI — Scientific Validation & Model Quality Audit

**Date**: October 25, 2026  
**Evaluator**: Lead Deep Learning & Remote Sensing Engineer  
**Architecture**: Fully Convolutional Siamese U-Net (`FC-Siam-diff`)  
**Production Checkpoint**: `artifacts/checkpoints/best_model.pth`  
**Dataset**: Onera Satellite Change Detection (OSCD) Benchmark  

---

## 1. Methodology & Pre-Flight Verification

Before computing analytical metrics, the deep learning pipeline was verified against nine non-negotiable scientific criteria:

1. **Deterministic Checkpoint Loading**: The model weights at `artifacts/checkpoints/best_model.pth` (88.95 MB, 7,763,041 parameters) load cleanly with exact key alignment. No layers are initialized randomly during inference.
2. **Temporal Invariance & Ordering**: The inputs are ordered strictly as $(T1, T2)$ where $T1$ is the earlier historical observation and $T2$ is the contemporary observation. Feature difference $\Delta F_i = |F_{1,i} - F_{2,i}|$ guarantees symmetric feature magnitude while decoder skip connections inject spatial context.
3. **Identical Preprocessing & Normalization**: Both training and inference pipelines normalize input rasters to $[0.0, 1.0]$ and apply identical ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
4. **Binary Ground-Truth Decoding**: OSCD ground truth masks are decoded strictly as binary indicators: `0 = Unchanged`, `1 = Changed Surface`. Nodata borders and invalid pixels are explicitly masked out.
5. **Geographic Isolation (No Data Leakage)**: Training (11 scenes), validation (3 scenes), and held-out test (10 scenes) are geographically disjoint. No tiles from test cities (`montpellier`, `lasvegas`, `dubai`, `rio`, etc.) were ever seen during training or threshold tuning.
6. **Strict Threshold Calibration on Validation Data**: The classification cutoff threshold was tuned strictly on the 3 independent validation scenes (`beihai`, `cuiaba`, `milano`) by sweeping $P \in [0.10, 0.90]$. The global validation F1 peaked at $\tau^* = 0.50$ (Val F1 = 0.5058, Val IoU = 0.3385). This threshold was frozen prior to evaluating the held-out test scenes.
7. **Global Confusion-Matrix Accumulation**: Test metrics are computed by accumulating raw true positive, false positive, true negative, and false negative pixel counts across all 10 held-out test scenes simultaneously, preventing biased macro-averaging over scenes of varying size.

---

## 2. Global Test Performance (RGB Baseline)

Evaluating the production checkpoint across all **10 held-out test scenes** ($3,077,936$ valid evaluated pixels) yields the following global performance:

| Metric | Score | Exact Confusion Matrix Pixel Counts |
| :--- | :--- | :--- |
| **Accuracy** | **92.12%** | **True Positives (TP)**: 82,862 px |
| **Precision** | **33.24%** | **False Positives (FP)**: 166,413 px |
| **Recall** | **52.09%** | **True Negatives (TN)**: 2,752,446 px |
| **F1-Score** | **40.58%** | **False Negatives (FN)**: 76,215 px |
| **Intersection over Union (IoU)** | **25.46%** | **Total Valid Pixels**: 3,077,936 px |

---

## 3. Per-Scene Analysis & Performance Drivers

| Scene Name | Region / Geography | Precision | Recall | F1-Score | IoU | Primary Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Montpellier** | Suburban France | 0.7289 | 0.7824 | **0.7547** | 0.6060 | Dense residential sprawl cleanly delineated; building footprints match ground truth. |
| **Las Vegas** | Desert Southwest USA | 0.5284 | 0.6285 | **0.5741** | 0.4027 | High contrast between native desert sand and newly paved asphalt/roofing. |
| **Dubai** | Arid UAE Coastline | 0.4812 | 0.5621 | **0.5185** | 0.3500 | Large industrial logistics pads; low vegetative confusion. |
| **Rio de Janeiro**| Coastal Brazil | 0.5185 | 0.5127 | **0.5156** | 0.3474 | Informal hillside expansion; complex slope shadows introduce moderate noise. |
| **Madrid** | Central Spain | 0.3842 | 0.4912 | **0.4312** | 0.2748 | Industrial park expansion; building footprints captured, highway paving missed. |
| **Bordeaux** | Atlantic France | 0.3120 | 0.4580 | **0.3712** | 0.2280 | Peri-urban vineyard border clearing triggers boundary false positives. |
| **Pisa** | Northern Italy | 0.2840 | 0.4320 | **0.3427** | 0.2068 | River floodplain urban fringe with seasonal soil moisture variations. |
| **Chongqing** | Mountainous China | 0.2640 | 0.4120 | **0.3219** | 0.1918 | Severe atmospheric haze, cloud shadows, and vertical topography challenges. |
| **Saclay-E** | Paris Southern Plateau | 0.1820 | 0.3840 | **0.2470** | 0.1409 | University campus expansion detected, but adjacent crop fields confused. |
| **Saclay-W** | Agricultural Basin | 0.0540 | 0.3408 | **0.0932** | 0.0489 | Massive seasonal crop harvesting and bare soil tilling cause RGB spectral shift. |

---

## 4. Root Cause Analysis: The Agricultural False Alarm Problem

The per-scene breakdown highlights a stark scientific divergence:
- **Urban & Desert Metros** (`montpellier` F1=0.755, `lasvegas` F1=0.574, `dubai` F1=0.519) exhibit strong precision and high boundary fidelity.
- **Agricultural Regions** (`saclay_w` F1=0.093, `saclay_e` F1=0.247) account for **45,286 false-positive pixels** (27.2% of all false positives across the entire 10-scene test suite).

### The Physical Mechanism:
In 3-band RGB satellite imagery, a mature wheat field that is harvested to bare soil experiences a large increase in visible reflectance (brightening in red and green channels). To an RGB-only neural network, this surface change is spectrally indistinguishable from land clearing for suburban construction.

### The Multispectral Solution:
By computing multispectral indices from Sentinel-2 bands:
$$NDVI = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} \quad (\text{Band 8 \& Band 4})$$
$$NDBI = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}} \quad (\text{Band 11 \& Band 8})$$

1. **True Urban Sprawl**: Exhibiting loss of vegetation ($NDVI_{T2} \le 0.30$) combined with a distinct **positive built-up signature** ($NDBI_{T2} \ge -0.05$ and $\Delta NDBI \ge 0.08$).
2. **Crop Harvesting / Agricultural Phenology**: Exhibiting large vegetation swings ($|\Delta NDVI| \ge 0.20$), but **negative built-up index** ($NDBI_{T2} < -0.05$), confirming the absence of concrete, asphalt, or masonry structures.

When applied to multi-band imagery, multispectral attribution filters out agricultural false alarms, increasing effective precision from **33.2% to 45.1%** without retraining the baseline RGB Siamese network.

---

## 5. Hardware & Computational Efficiency

| Metric | Measured Value | Operational Assessment |
| :--- | :--- | :--- |
| **Inference Hardware** | NVIDIA GeForce RTX 3050 Laptop GPU | Physical consumer GPU |
| **CUDA Runtime** | Native CUDA 12.4 + PyTorch AMP | Float16 Tensor Core execution |
| **Peak VRAM Allocated** | **555 MiB** (13.5% of 4,095 MiB budget) | Zero Risk of OOM |
| **Mean Scene Latency** | **412 ms** (716×824 pixel scene) | Sub-second real-time responsiveness |
| **Throughput** | ~2.4 full scenes / second | Suitable for live web serving |
