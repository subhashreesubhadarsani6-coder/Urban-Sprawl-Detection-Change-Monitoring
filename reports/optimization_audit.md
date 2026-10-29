# Project & Baseline Model Optimization Audit

## 1. Executive Summary

This audit rigorously inspects the source code, dataset, configuration, checkpoints, logs, and evaluation reports of the **UrbanScope AI** change detection system.

- **Model**: Fully Convolutional Siamese Difference U-Net (`FC-Siam-diff`)
- **Parameters**: 7,763,041 (7.76M)
- **Baseline Checkpoint**: `artifacts/checkpoints/best_model.pth` (88.95 MB, Epoch 10)
- **Backup Location**: `artifacts/backups/best_model.pth` and `artifacts/backups/last_checkpoint.pth`
- **Audit Date**: 2026-10-25

---

## 2. Verification of the 10 Audit Items

| Item | Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :--- |
| **1** | Checkpoint files exist and can be loaded | `artifacts/checkpoints/best_model.pth` exists, is 88.95 MB, loads cleanly with PyTorch. Contains state_dict, optimizer, scheduler, RNG seeds, and training history. | **VERIFIED** |
| **2** | Model architecture matches checkpoint | Strict state dict key loading confirmed with 0 missing and 0 unexpected keys. Exactly 7,763,041 parameters. | **VERIFIED** |
| **3** | Identical normalization in train & inference | Both pipelines utilize ImageNet RGB normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$). Verified numerically identical. | **VERIFIED** |
| **4** | Correct T1/T2 image ordering | In OSCD Parquet shards, `image1` is the earlier acquisition and `image2` is the later acquisition. Formats verified across dataset reader and sliding window inference. | **VERIFIED** |
| **5** | Ground-truth masks correctly decoded | Mask bytes converted via PIL and thresholded strictly at $>0$ to form integer binary $\{0, 1\}$ ground truth. | **VERIFIED** |
| **6** | Nodata pixels excluded consistently | Pixels where all RGB channels equal 0 in both T1 and T2 are flagged invalid and masked out from BCE and Soft Dice losses and all evaluation metrics. | **VERIFIED** |
| **7** | Geographic split isolation | 11 train, 3 validation, and 10 test scenes are completely mutually exclusive. Set intersection is strictly $\emptyset$, guaranteeing zero spatial data leakage. | **VERIFIED** |
| **8** | Global metric accumulation | `ChangeDetectionMetrics` accumulates global TP, FP, TN, and FN across the entire dataset before calculating Precision, Recall, F1, and IoU, eliminating batch-averaging bias. | **VERIFIED** |
| **9** | Reproduction of reported scores | Re-evaluated baseline checkpoint on validation split: Reproduced **Val F1: 0.5058**, **Val IoU: 0.3385**, exactly matching saved checkpoint metrics. | **VERIFIED** |
| **10** | Metrics authenticity in API / UI | SQLite database `backend/urbanscope.db` holds real inference metrics from FC-Siam-diff for Las Vegas ($60,613$ changed pixels at $0.50$ cutoff; $6.0613\text{ km}^2$). | **VERIFIED** |

---

## 3. Baseline Metric Benchmarks

- **Validation F1**: `0.5058`
- **Validation IoU**: `0.3385`
- **Validation Precision**: `0.4583`
- **Validation Recall**: `0.5644`
- **Held-out Test F1**: `0.4058`
- **Held-out Test IoU**: `0.2546`
- **Held-out Test Precision**: `0.3324`
- **Held-out Test Recall**: `0.5209`
- **Held-out Test Accuracy**: `92.12%`
