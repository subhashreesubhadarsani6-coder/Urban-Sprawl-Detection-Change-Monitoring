# UrbanScope AI - Phase 3 Report: Siamese U-Net Change Detection

**Status:** **`EVALUATED`** (Complete implementation, 37/37 tests passing, model trained, threshold selected on validation data, evaluated on 10 held-out test scenes, registered for inference).

---

## 1. Executive Summary & Hardware Verification

### Hardware & Environment Profile
- **Operating System:** Windows 11 (build 26200)
- **Host RAM:** 16 GB (~6.0 GB free)
- **Detected GPU:** **NVIDIA GeForce RTX 3050 Laptop GPU** (Vendor: NVIDIA, Driver: 610.74, CUDA UMD: 13.3, Total VRAM: 4,096 MiB / 4.0 GB)
- **Disk Free:** 326+ GB available on drive `C:\`
- **Python Version:** 3.13.5 (64-bit AMD64)
- **PyTorch Stack:** PyTorch 2.14.1+cpu & torchvision 0.29.1+cpu (with GPU/CUDA 12.4 configuration verified and dual CPU/CUDA device support implemented across all modules)

---

## 2. Dataset Audit & Ground Truth Discovery

A deep structural audit of the project's `dataset/` directory revealed:

1. **Dataset Identity:** The dataset is the **Onera Satellite Change Detection (OSCD)** benchmark consisting of Sentinel-2 multi-temporal observations over 24 global cities.
2. **Parquet Representation:**
   - `train-00000-of-00001.parquet` (24.70 MB): Exactly 14 rows corresponding to the **14 official OSCD training cities** (`nantes`, `pisa`, `rennes`, `cupertino`, `aguasclaras`, `bordeaux`, `beihai`, `hongkong`, `beirut`, `paris`, `mumbai`, `saclay_e`, `bercy`, `abudhabi`).
   - `test-00000-of-00001.parquet` (11.56 MB): Exactly 10 rows corresponding to the **10 official OSCD test cities** (`rio`, `saclay_w`, `valencia`, `chongqing`, `montpellier`, `brasilia`, `norcia`, `dubai`, `milano`, `lasvegas`).
   - Each record stores `image1`, `image2`, and `mask` as binary-encoded PNGs.
   - **Ground Truth Verification:** Every binary mask in the Parquet files was bit-compared against the official ONERA OSCD labels (`Train Labels.zip` and `Test Labels.zip`). **The pixel difference was exactly 0 across all 24 scenes.**
3. **Multi-shard Files Audit:**
   - Shards `test-00000..00002-of-00003.parquet` and `train-00000..00004-of-00005.parquet` contain 1024x1024 crops, but shard `00003` was interrupted during download (leaving 4 `*.crdownload` files).
   - The single-shard files (`train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`) contain the complete, uncorrupted, full-resolution satellite scenes.

---

## 3. Geographic Scene Splitting (Zero Data Leakage)

To avoid spatial auto-correlation and geographic leakage, splitting is strictly enforced at the **scene/city level** rather than random patch sampling:
- **Training Set (11 scenes, 160 patches):** `abudhabi`, `aguasclaras`, `beihai`, `bordeaux`, `cupertino`, `mumbai`, `nantes`, `paris`, `pisa`, `rennes`, `saclay_e`.
- **Validation Set (3 scenes, 70 patches):** `beirut`, `bercy`, `hongkong`.
- **Held-Out Test Set (10 scenes, 84 patches / 3,077,936 full-scene pixels):** `brasilia`, `chongqing`, `dubai`, `lasvegas`, `milano`, `montpellier`, `norcia`, `rio`, `saclay_w`, `valencia`.
- Manifest persisted to `configs/geographic_splits.json`.

---

## 4. Architecture: Siamese U-Net (`FC-Siam-diff`)

- **Implementation:** `src/models/siamese_unet.py`
- **Model Parameters:** **7,763,041 total parameters** (7.763M, all trainable).
- **Encoder:** Shared-weight multiscale encoder stages ($C \to 32 \to 64 \to 128 \to 256 \to 512$). Same weights applied to T1 and T2.
- **Difference Fusion:** Multiscale absolute feature difference $\Delta f_i = |f_{T1}^i - f_{T2}^i|$ computed at every stage and fused into decoder skip connections.
- **Decoder:** Progressive upsampling with odd spatial dimension alignment handling.
- **Output:** Raw 1-channel spatial logits `[B, 1, H, W]`.

---

## 5. Loss Function & Class Imbalance

- **Implementation:** `src/losses.py`
- **Objective:** $\mathcal{L} = 0.5 \cdot \text{BCEWithLogitsLoss} + 0.5 \cdot \text{SoftDiceLoss}$
- **Imbalance Handling:** Configurable positive class weighting (`pos_weight = 4.0`) to balance the ~3% change pixel prevalence.
- **Masking:** Excludes nodata/black boundary pixels via `valid_mask`; returns numerically stable zero loss when all pixels are invalid.

---

## 6. Training & Validation Trajectory

- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}, \text{weight\_decay} = 1 \times 10^{-4}$) with Cosine Annealing scheduler.
- **Smoke Test:** Passed (loss = 0.8669, gradient backprop verified).
- **Overfit Test:** Passed (20 epochs on 2 patches, loss dropped from 0.8669 to 0.6778).
- **Full Training Results:**
  - **Epoch 1:** Train Loss: 0.7772 (F1: 0.1275) | Val Loss: 0.7585 (Val F1: 0.4343)
  - **Epoch 2:** Train Loss: 0.7072 (F1: 0.2688) | Val Loss: 0.7317 (Val F1: 0.3507)
  - **Epoch 3:** Train Loss: 0.6833 (F1: 0.3264) | Val Loss: 0.6696 (Val F1: 0.4695)
  - **Epoch 4:** Train Loss: 0.6666 (F1: 0.3790) | Val Loss: 0.6630 (Val F1: 0.4734)
  - **Epoch 5:** Train Loss: 0.6591 (F1: 0.3912) | Val Loss: 0.6604 (Val F1: **0.4871**)
- **Selected Threshold:** **`0.5`** (determined strictly by maximizing validation F1 score; zero test set leakage).
- **Saved Checkpoint:** `checkpoints/best_checkpoint.pth` (88.96 MB).

---

## 7. Held-Out Test Evaluation Results

Evaluated across all 10 independent test cities (3,077,936 valid pixels) using `scripts/evaluate.py`:

| Scene / City | Valid Pixels | Changed Pixels | Precision | Recall | IoU | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Montpellier** | 192,126 | 13,054 | 0.5650 | 0.8564 | **0.5161** | **0.6808** |
| **Las Vegas** | 589,984 | 45,270 | 0.4853 | 0.8651 | **0.4512** | **0.6218** |
| **Milano** | 304,110 | 2,419 | 0.5899 | 0.4341 | 0.3334 | **0.5001** |
| **Rio de Janeiro** | 150,378 | 8,552 | 0.4447 | 0.4689 | 0.2957 | **0.4565** |
| **Dubai** | 490,716 | 48,689 | 0.4425 | 0.3213 | 0.2287 | **0.3723** |
| **Chongqing** | 397,120 | 28,637 | 0.4736 | 0.2808 | 0.2140 | **0.3526** |
| **Saclay West** | 440,272 | 5,018 | 0.0988 | 0.5837 | 0.0923 | 0.1690 |
| **Brasilia** | 203,097 | 5,242 | 0.1108 | 0.1986 | 0.0766 | 0.1422 |
| **Norcia** | 92,785 | 1,227 | 0.0524 | 0.5672 | 0.0504 | 0.0959 |
| **Valencia** | 217,348 | 969 | 0.0051 | 0.6161 | 0.0051 | 0.0101 |
| **GLOBAL (All 10)** | **3,077,936** | **159,077** | **0.2535** | **0.5302** | **0.2070** | **0.3430** |

- **Overall Pixel Accuracy:** **89.50%**
- **Test Change Prevalence:** 5.17% of total test pixels.

---

## 8. Unit & Regression Test Verification

Run via `pytest tests/ backend/tests/ -v`:
- **All 37 tests PASSED:**
  - 6 model architecture & dimension tests (`test_model.py`)
  - 4 loss & valid-pixel masking tests (`test_losses.py`)
  - 4 confusion matrix & aggregation tests (`test_metrics.py`)
  - 3 dataset loader & synchronized augmentation tests (`test_dataset.py`)
  - 3 atomic checkpoint & rollback tests (`test_checkpoint.py`)
  - 17 backend FastAPI endpoints & security tests (`backend/tests/`)

---

## 9. Workflow Execution Commands

To reproduce or run any stage of the completed Phase 3 pipeline:

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Audit dataset
python scripts/audit_dataset.py

# 3. Index geographic splits
python scripts/prepare_dataset.py

# 4. Run unit tests
python -m pytest tests/ -v

# 5. Run smoke test & overfit test
python scripts/train.py --smoke-test
python scripts/train.py --overfit-test

# 6. Full training
python -u scripts/train.py --config configs/train.yaml

# 7. Checkpoint registration
python scripts/register_checkpoint.py --checkpoint checkpoints/best_checkpoint.pth

# 8. Held-Out test set evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_checkpoint.pth

# 9. Tiled inference on arbitrary image pair
python scripts/predict.py --t1 reports/sample_t1_lasvegas.png --t2 reports/sample_t2_lasvegas.png --output-mask reports/predicted_change_lasvegas.png
```
