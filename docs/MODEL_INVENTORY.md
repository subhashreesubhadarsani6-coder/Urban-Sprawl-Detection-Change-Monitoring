# UrbanScope AI — Machine Learning & Deep Learning Model Inventory

**Audit Date**: October 29, 2026  
**Auditor**: Senior Deep Learning Engineer & Remote Sensing Specialist  
**Project Path**: `C:\Project\Urban Sprawl Detection & Change Monitoring`  
**Execution Context**: Windows 11 Enterprise, Python 3.13.5, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM)  

---

## 1. Executive Model Status Matrix

To avoid ambiguous claims, every model mentioned across the codebase, documentation, and user interface is classified into one of six verifiable stages:

| Model / Component Identifier | Mentioned in Docs | Defined in Source | Checkpoint Exists | Actually Trained | Evaluated on Test Set | Integrated in Live App | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`FC-Siam-diff` (Production Baseline)** | Yes | Yes | Yes (88.95 MB) | Yes (18 epochs) | Yes (10 test cities) | **Yes (Active)** | **Production Primary** |
| **`FC-Siam-diff` (exp0_baseline)** | Yes | Yes | Yes (88.95 MB) | Yes (12 epochs) | Yes (Val F1=0.4415) | No | Experimental Benchmark |
| **`FC-Siam-diff` (exp1_balanced_posweight)** | Yes | Yes | Yes (88.95 MB) | Yes (9 epochs) | Yes (Val F1=0.4483) | No | Experimental Benchmark |
| **`FC-Siam-diff` (exp2_dice_dominant)** | Yes | Yes | Yes (88.95 MB) | Yes (12 epochs) | Yes (Val F1=0.4420) | No | Experimental Benchmark |
| **`FC-Siam-diff` (exp3_dense_stride)** | Yes | Yes | Yes (88.95 MB) | Yes (10 epochs) | Yes (Val F1=0.4897) | No | Experimental Benchmark |
| **`FC-Siam-diff` (exp4_optimized)** | Yes | Yes | Yes (88.95 MB) | Yes (18 epochs) | Yes (Val F1=0.5058) | Yes (Backup sync) | Optimization Checkpoint |
| **Multispectral Attribution (NDVI/NDBI)** | Yes | Yes | N/A (Rule Engine) | N/A (Algorithmic) | Yes (Synthetic & Real) | **Yes (Active)** | **Active Post-Processor** |
| **`ChangeFormer` (Siamese Transformer)** | Yes (UI) | No | No | No | No | No | **UI Mock / Benchmark Only** |
| **`BIT-CD` (Bitemporal Transformer)** | Yes (UI) | No | No | No | No | No | **UI Mock / Benchmark Only** |
| **`Siam-UNet` (Siamese Concatenation)** | Yes (UI) | No | No | No | No | No | **UI Mock / Benchmark Only** |

---

## 2. Primary Production Model: FC-Siam-diff

### 2.1 Architectural Specifications
- **Model Name**: Fully Convolutional Siamese Difference U-Net (`FC-Siam-diff`)
- **Primary Source Code Files**:
  - [`models/siam_unet.py`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/models/siam_unet.py#L56-L185) (`class FCSiamDiff(nn.Module)`)
  - [`src/models/siamese_unet.py`](file:///C:/Project/Urban%20Sprawl%20Detection%20&%20Change%20Monitoring/src/models/siamese_unet.py#L53-L173) (`class FCSiamDiff(nn.Module)`)
- **Model Family**: Dual-branch Siamese Fully Convolutional Network with encoder-decoder skip connections.
- **Task Type**: Bi-temporal pixel-level semantic change segmentation / binary change detection.
- **Framework**: PyTorch (`torch>=2.2.0`, currently executing on native `torch==2.6.0+cu124`).
- **Total Trainable Parameters**: **7,763,041** parameters (31.05 MB uncompressed FP32).
- **Weight Sharing**: Shared Siamese Encoder. Exactly one instance of encoder weights extracts multiscale representations for both $T1$ and $T2$, guaranteeing temporal feature symmetry.

### 2.2 Layer Topology & Tensor Dimension Progression
The network consists of a shared 4-stage multiscale encoder, a feature difference operator at each scale, and a 4-stage decoder:

```
T1 Image [B, 3, H, W] ---> [Shared Encoder Block 1] ---> f1_1 [B, 32, H, W]   -----> (Diff 1) |f1_1 - f2_1| [B, 32, H, W] --------+
T2 Image [B, 3, H, W] ---> [Shared Encoder Block 1] ---> f2_1 [B, 32, H, W]                                                        |
                                  |                                                                                                  |
                            [MaxPool 2x2]                                                                                            |
                                  v                                                                                                  |
                           [Shared Encoder Block 2] ---> f1_2, f2_2 [B, 64, H/2, W/2] -> (Diff 2) [B, 64, H/2, W/2] -------+      |
                                  |                                                                                         |      |
                            [MaxPool 2x2]                                                                                   |      |
                                  v                                                                                         |      |
                           [Shared Encoder Block 3] ---> f1_3, f2_3 [B, 128, H/4, W/4] -> (Diff 3) [B, 128, H/4, W/4] -+   |      |
                                  |                                                                                    |   |      |
                            [MaxPool 2x2]                                                                              |   |      |
                                  v                                                                                    |   |      |
                           [Shared Encoder Block 4] ---> f1_4, f2_4 [B, 256, H/8, W/8] -> (Diff 4) [B, 256, H/8, W/8] |   |      |
                                  |                                                                                    |   |      |
                            [MaxPool 2x2]                                                                              |   |      |
                                  v                                                                                    |   |      |
                           [Shared Bottleneck Block] --> x [B, 512, H/16, W/16]                                        |   |      |
                                  |                                                                                    |   |      |
                                  v                                                                                    |   |      |
                           [Decoder Block 4] <=========================================================================+   |      |
                             (ConvTranspose2d 2x2 + Cat[Skip Diff 4] + DoubleConv) -> [B, 256, H/8, W/8]                   |      |
                                  |                                                                                        |      |
                                  v                                                                                        |      |
                           [Decoder Block 3] <=============================================================================+      |
                             (ConvTranspose2d 2x2 + Cat[Skip Diff 3] + DoubleConv) -> [B, 128, H/4, W/4]                          |
                                  |                                                                                               |
                                  v                                                                                               |
                           [Decoder Block 2] <====================================================================================+
                             (ConvTranspose2d 2x2 + Cat[Skip Diff 2] + DoubleConv) -> [B, 64, H/2, W/2]
                                  |
                                  v
                           [Decoder Block 1] <==================================================================================== Skip Diff 1
                             (ConvTranspose2d 2x2 + Cat[Skip Diff 1] + DoubleConv) -> [B, 32, H, W]
                                  |
                                  v
                           [Classifier Head: Conv2d(32, 1, kernel_size=1)]
                                  |
                                  v
                           Raw Change Logits [B, 1, H, W]
```

### 2.3 Input and Output Contracts
- **Input Tensors**:
  - `t1`: Tensor of shape `[Batch_Size, 3, Height, Width]`, dtype `torch.float32`.
  - `t2`: Tensor of shape `[Batch_Size, 3, Height, Width]`, dtype `torch.float32`.
  - Image channels correspond to RGB reflectance normalized to $[0.0, 1.0]$ and standardized via ImageNet distribution ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Output Tensor**:
  - `logits`: Tensor of shape `[Batch_Size, 1, Height, Width]`, dtype `torch.float32`.
  - Continuous unconstrained logits where $\sigma(\text{logit}) \in [0.0, 1.0]$ represents pixel-wise change probability.

---

## 3. Loss Functions & Optimization Strategy

### 3.1 Loss Formulation
The loss function is implemented in [`src/losses.py`](file:///C:/Project/Urban Sprawl Detection & Change Monitoring/src/losses.py#L48-L113) (`class CombinedLoss(nn.Module)`):
$$\mathcal{L}_{\text{total}} = w_{\text{bce}} \cdot \mathcal{L}_{\text{BCE}} + w_{\text{dice}} \cdot \mathcal{L}_{\text{Dice}}$$

1. **Weighted Binary Cross Entropy (`BCEWithLogitsLoss`)**:
   $$\mathcal{L}_{\text{BCE}}(p, y) = - \frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \log \sigma(p_i) + (1 - y_i) \log(1 - \sigma(p_i)) \right]$$
   - $w_{\text{pos}} = 3.0$ (production checkpoint) or $5.0$ (baseline). Penalizes false negatives to counteract extreme class imbalance (urban change occupies $< 5\%$ of satellite scene pixels).
2. **Soft Dice Loss**:
   $$\mathcal{L}_{\text{Dice}}(p, y) = 1 - \frac{2 \sum_i \sigma(p_i) y_i + \epsilon}{\sum_i \sigma(p_i) + \sum_i y_i + \epsilon}$$
   - Direct spatial overlap optimization; enforces structural coherence and penalizes salt-and-pepper noise.
3. **Nodata & Border Mask Exclusion**:
   - Both BCE and Soft Dice terms strictly evaluate over valid pixels ($M_{\text{valid}} = 1$), excluding black satellite padding borders and nodata fill values.

### 3.2 Optimization & Scheduling
- **Optimizer**: `AdamW` (`lr = 1e-4`, `weight_decay = 0.01`, `betas = (0.9, 0.999)`).
- **Learning Rate Scheduler**: `CosineAnnealingLR` ($T_{\max} = 40$, $\eta_{\min} = 10^{-6}$).
- **Gradient Clipping**: Norm clipped at `max_norm = 1.0` to avoid exploding gradients during mixed precision backpropagation.
- **Hardware Acceleration**: Automatic Mixed Precision (`torch.amp.autocast('cuda')` with `torch.amp.GradScaler('cuda')`).

---

## 4. Inference & Tiling Engine

Implemented in [`src/inference/predictor.py`](file:///C:/Project/Urban Sprawl Detection & Change Monitoring/src/inference/predictor.py#L25-L162) and [`backend/app/services/geospatial_engine.py`](file:///C:/Project/Urban Sprawl Detection & Change Monitoring/backend/app/services/geospatial_engine.py#L320-L360):

1. **Sliding-Window Patch Extraction**:
   - Arbitrary satellite scenes are divided into $256 \times 256$ patches with a $64 \text{ px}$ overlap ($25\%$ spatial overlap).
2. **2D Hann Window Blending**:
   - Patch outputs are weighted by a 2D Hann window:
     $$W(x, y) = \sin^2\left(\frac{\pi x}{P}\right) \cdot \sin^2\left(\frac{\pi y}{P}\right)$$
   - Reconstructed probability map:
     $$P_{\text{scene}}(x, y) = \frac{\sum_{k} W_k(x, y) \cdot P_k(x, y)}{\sum_k W_k(x, y)}$$
   - Eliminates hard tile border seams and discontinuity artifacts across window boundaries.
3. **Dynamic Re-Thresholding**:
   - The continuous probability map is cached in backend memory and persisted as a 16-bit PNG.
   - Adjusting the cutoff slider on the frontend triggers instant sub-100ms binary mask re-vectorization via `rethreshold_saved_job()` without re-executing neural network forward passes.

---

## 5. Multispectral Attribution Algorithm

Implemented in [`backend/app/geospatial/spectral_indices.py`](file:///C:/Project/Urban Sprawl Detection & Change Monitoring/backend/app/geospatial/spectral_indices.py#L140-L290):

### 5.1 Formulation
- **NDVI** (Normalized Difference Vegetation Index):
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{B_{08} - B_{04}}{B_{08} + B_{04}}$$
- **NDBI** (Normalized Difference Built-up Index):
  $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}} = \frac{B_{11} - B_{08}}{B_{11} + B_{08}}$$
- **MNDWI** (Modified Normalized Difference Water Index):
  $$\text{MNDWI} = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}} = \frac{B_{03} - B_{11}}{B_{03} + B_{11}}$$

### 5.2 Attribution Decision Rules
Given a candidate pixel identified as changed by the RGB Siamese U-Net ($P(\text{change}) \ge \tau$):

1. **Class 1 — Verified Urban Sprawl**:
   $$\text{Condition: } (\Delta\text{NDBI} \ge 0.08 \text{ or } \text{NDBI}_{T2} \ge -0.05) \land (\text{NDVI}_{T2} \le 0.30)$$
   *Physical Meaning*: Soil or vegetative land replaced by concrete, asphalt, roofing, or compacted impervious ground.
2. **Class 2 — Agricultural Phenology / Crop Cycle**:
   $$\text{Condition: } (|\Delta\text{NDVI}| \ge 0.20 \text{ or } \text{NDVI}_{T1} > 0.35) \land (\text{NDBI}_{T2} < -0.05)$$
   *Physical Meaning*: Crop harvest, haying, plowing, or seasonal green-up where visible RGB reflectance brightens sharply but no built-up spectral signature is present.
3. **Class 3 — Water Dynamics**:
   $$\text{Condition: } \text{MNDWI}_{T2} > 0.10 \lor \Delta\text{MNDWI} > 0.25$$
   *Physical Meaning*: Reservoir fluctuations, retention pond filling, or river sedimentation.
4. **Class 4 — Bare Soil Excavation**:
   $$\text{Condition: } \text{NDVI}_{T2} < 0.22 \land -0.15 \le \text{NDBI}_{T2} < -0.05$$
   *Physical Meaning*: Earthmoving, grading, or site clearing prior to structural concrete pouring.

---

## 6. Resource Consumption & Operational Limits

| Resource Parameter | Measured Value | Practical Constraint |
| :--- | :--- | :--- |
| **Peak GPU VRAM Allocated** | **555 MiB** | Fits easily inside 4 GB VRAM budget on NVIDIA RTX 3050 Laptop GPU |
| **CPU RAM Consumption** | ~680 MB | Fits easily on systems with 8 GB to 16 GB RAM |
| **GPU Inference Latency** | **412 ms** per $716 \times 824$ scene | Real-time web responsiveness |
| **CPU Fallback Latency** | ~35 seconds per $716 \times 824$ scene | Usable offline, but too slow for interactive slider dragging |
| **Disk Storage per Checkpoint**| **88.95 MB** (`.pth` file) | Light storage footprint |
| **Disk Storage per Audit Run** | ~4.5 MB (GeoTIFF, GeoJSON, PNGs) | Stored in `backend/storage/artifacts/{job_id}/` |
