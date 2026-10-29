# Spatial Error Analysis & Prediction Audit Report

## 1. Executive Summary

This report delivers a spatial, radiometric, and categorical decomposition of model predictions across all 10 independent held-out test scenes from the Onera Satellite Change Detection (OSCD) dataset.

- **Baseline Checkpoint**: `artifacts/checkpoints/best_model.pth`
- **Optimal Probability Cutoff**: `0.50` (Tuned strictly on validation scenes)
- **Global Test Accuracy**: `92.12%`
- **Global Test Metrics**: Precision = `0.3324`, Recall = `0.5209`, F1 = `0.4058`, IoU = `0.2546`
- **Total Evaluated Pixels**: `3,077,936` valid pixels
- **Confusion Matrix**: TP = `82,862` | FP = `166,413` | TN = `2,752,446` | FN = `76,215`

---

## 2. Per-Scene Performance Ranking

Scenes ranked by test F1-score from best to worst:

| Rank | Scene ID | Geographic Region | Test F1 | Test IoU | Precision | Recall | Actual Changed Px | False Positives | False Negatives |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **montpellier** | France (Mediterranean) | **0.7547** | **0.6061** | 0.7095 | 0.8061 | 13,054 | 4,309 | 2,531 |
| 2 | **lasvegas** | USA (Desert Southwest) | **0.5741** | **0.4026** | 0.4312 | 0.8585 | 45,270 | 51,259 | 6,404 |
| 3 | **rio** | Brazil (Coastal Urban) | **0.5156** | **0.3474** | 0.5185 | 0.5127 | 8,552 | 4,072 | 4,167 |
| 4 | **dubai** | UAE (Arid Coastal) | **0.4003** | **0.2502** | 0.4107 | 0.3904 | 48,689 | 27,278 | 29,679 |
| 5 | **brasilia** | Brazil (Cerrado Plateau) | **0.2829** | **0.1648** | 0.2799 | 0.2860 | 5,242 | 3,856 | 3,743 |
| 6 | **chongqing** | China (River Basin) | **0.2684** | **0.1550** | 0.4495 | 0.1913 | 28,637 | 6,711 | 23,158 |
| 7 | **milano** | Italy (Po Valley Plain) | **0.2497** | **0.1427** | 0.3563 | 0.1922 | 2,419 | 840 | 1,954 |
| 8 | **saclay_w** | France (Agricultural Plain) | **0.0932** | **0.0489** | 0.0540 | 0.3408 | 5,018 | 29,966 | 3,308 |
| 9 | **norcia** | Italy (Apennine Mountain) | **0.0787** | **0.0410** | 0.0436 | 0.4059 | 1,227 | 10,928 | 729 |
| 10 | **valencia** | Spain (Coastal Plain) | **0.0299** | **0.0152** | 0.0155 | 0.4407 | 969 | 27,194 | 542 |

---

## 3. Root Cause Analysis of Prediction Errors

Visual and radiometric inspection of the 8-panel diagnostic figures in `reports/optimization/predictions/` reveals three distinct failure modes:

### A. Agricultural Phenology & Crop Harvests (Valencia, Saclay-W)
- **Observation**: In `valencia` (F1 = 0.0299) and `saclay_w` (F1 = 0.0932), the ground-truth change is minimal (969 px and 5,018 px), yet the model predicted 27,621 and 31,676 positive pixels.
- **Physical Cause**: Agricultural parcels transitioning between vegetated crop canopy (green) and plowed bare earth (brown/tan). The feature difference network $D_i = |F_{T1} - F_{T2}|$ generates high activation across entire agricultural plots.
- **Remedy**: Enhancing the loss function with spatial boundary regularizers and tuning positive class weighting to avoid over-predicting seasonal spectral shifts as structural urban sprawl.

### B. Topographic Relief & Sun Elevation Shadows (Norcia)
- **Observation**: `norcia` exhibits severe false alarms on mountain ridge slopes facing north/northwest (FP = 10,928 vs Actual Change = 1,227).
- **Physical Cause**: Steep mountainous terrain where differing solar azimuth and elevation between observation dates create deep topographic shadows.
- **Remedy**: Training with increased radiometric contrast jitter and normalized difference features.

### C. Atmospheric Haze & Contrast Attenuation (Chongqing)
- **Observation**: In `chongqing`, Recall is only 19.13% (23,158 missed change pixels).
- **Physical Cause**: Persistent basin cloud cover, particulate haze, and moisture that dampen contrast in the earlier observation T1.
- **Remedy**: Test-time and training contrast normalization and multi-band feature extraction.

---

## 4. Evidence-Based Optimization Recommendations

1. **Class-Balanced Loss Tuning**: Adjust `pos_weight` from 5.0 to 3.0 or 2.5 to penalize false positive alarms in rural/agricultural parcels.
2. **Hard-Negative Mining / Change-Aware Patch Sampling**: Ensure training patches contain a balanced ratio of positive-change scenes and challenging agricultural negative scenes.
3. **Probability Threshold Calibration**: Higher thresholds (e.g. 0.60–0.65) suppress seasonal field noise while preserving large urban expansion clusters.
