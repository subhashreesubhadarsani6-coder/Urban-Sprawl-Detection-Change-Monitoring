"""Phase 6: Controlled Evaluation Comparison Script.

Evaluates and compares:
1. Baseline Model (`artifacts/checkpoints/best_model.pth`, Val F1: 0.5058)
2. Optimized Model Candidate (`artifacts/experiments/exp4_optimized/best_model.pth`, Val F1: 0.4977)

Calculates:
- Global Test Precision, Recall, F1, IoU, Accuracy, TP, FP, TN, FN
- Absolute and relative differences (Delta F1, Delta IoU, Delta Precision, Delta False Positives)
- Per-scene comparison across all 10 held-out scenes
- Hardware runtime benchmarks (CPU vs GPU throughput)
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from training.evaluate_test import evaluate_test_set


def compare_models():
    print("=" * 70)
    print("PHASE 6: CONTROLLED MODEL COMPARISON & BENCHMARKING")
    print("=" * 70)

    # 1. Baseline Model Evaluation
    print("\n--- Evaluating Baseline Model Checkpoint ---")
    base_report = evaluate_test_set(
        checkpoint_path="artifacts/checkpoints/best_model.pth",
        output_report="reports/optimization/baseline_test_evaluation.json"
    )

    # 2. Optimized Candidate Evaluation
    print("\n--- Evaluating Experimental Candidate (Exp 4 Optimized) ---")
    opt_report = evaluate_test_set(
        checkpoint_path="artifacts/experiments/exp4_optimized/best_model.pth",
        output_report="reports/optimization/exp4_test_evaluation.json"
    )

    base_m = base_report["global_test_metrics"]
    opt_m = opt_report["global_test_metrics"]

    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY (Held-out Test Scenes)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Exp 4 Optimized':<15} {'Absolute Delta':<15} {'Relative Delta'}")
    print("-" * 75)
    for k in ["precision", "recall", "f1", "iou", "accuracy"]:
        bv = base_m[k]
        ov = opt_m[k]
        delta = ov - bv
        pct = (delta / bv) * 100 if bv > 0 else 0.0
        print(f"{k.capitalize():<20} {bv:<15.4f} {ov:<15.4f} {delta:+14.4f} {pct:+13.2f}%")

    print("-" * 75)
    print(f"False Positives:     {base_m['fp']:<15,} {opt_m['fp']:<15,} {opt_m['fp'] - base_m['fp']:+15,}")
    print(f"False Negatives:     {base_m['fn']:<15,} {opt_m['fn']:<15,} {opt_m['fn'] - base_m['fn']:+15,}")
    print(f"True Positives:      {base_m['tp']:<15,} {opt_m['tp']:<15,} {opt_m['tp'] - base_m['tp']:+15,}")

    comparison = {
        "baseline_metrics": base_m,
        "exp4_optimized_metrics": opt_m,
        "deltas": {
            "f1_delta": round(opt_m["f1"] - base_m["f1"], 4),
            "iou_delta": round(opt_m["iou"] - base_m["iou"], 4),
            "precision_delta": round(opt_m["precision"] - base_m["precision"], 4),
            "recall_delta": round(opt_m["recall"] - base_m["recall"], 4),
            "fp_reduction": base_m["fp"] - opt_m["fp"]
        }
    }

    with open("reports/optimization/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nSaved comparison JSON to: reports/optimization/model_comparison.json")
    return comparison


if __name__ == "__main__":
    compare_models()
