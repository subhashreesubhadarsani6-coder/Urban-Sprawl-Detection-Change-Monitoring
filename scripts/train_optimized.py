"""Train Optimized Candidate Model on RTX 3050 GPU.

Combines best experimental findings:
- Dense stride (160) for spatial coverage and negative parcel diversity (102 patches)
- Tuned pos_weight (3.0) to balance agricultural false-positives against structural change recall
- CosineAnnealingLR across 20 epochs with patience 8
- CUDA AMP enabled on RTX 3050 Laptop GPU

Stores checkpoint in: artifacts/experiments/exp4_optimized/best_model.pth
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath("."))
from scripts.run_experiments import run_experiment

if __name__ == "__main__":
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training Optimized Model on {device_str}...")
    res = run_experiment(
        exp_name="exp4_optimized",
        description="Optimized candidate (stride=160, pos_weight=3.0, bce=0.5, dice=0.5)",
        pos_weight=3.0,
        bce_weight=0.5,
        dice_weight=0.5,
        stride=160,
        epochs=20,
        patience=8,
        batch_size=4,
        lr=0.0001,
        seed=42,
        device_str=device_str
    )
    print("\nOptimized Candidate Training Result:")
    print(json.dumps(res, indent=2))
