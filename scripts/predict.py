"""Command-line Inference Script for Siamese U-Net Change Detection.

Accepts an earlier (T1) and later (T2) satellite image, executes tiled inference
with seamless blending, and writes binary change mask and probability maps.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from src.inference.predictor import ChangePredictor

def main():
    parser = argparse.ArgumentParser(description="Predict change between two satellite images")
    parser.add_argument("--t1", required=True, help="Path to earlier observation (T1) image")
    parser.add_argument("--t2", required=True, help="Path to later observation (T2) image")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pth", help="Path to checkpoint")
    parser.add_argument("--output-mask", default="reports/prediction_mask.png", help="Path to save binary change mask")
    parser.add_argument("--output-prob", default=None, help="Optional path to save probability map")
    parser.add_argument("--threshold", type=float, default=None, help="Classification probability threshold")
    parser.add_argument("--patch-size", type=int, default=256, help="Tiling patch size")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels")
    args = parser.parse_args()

    assert os.path.exists(args.t1), f"T1 image file not found: {args.t1}"
    assert os.path.exists(args.t2), f"T2 image file not found: {args.t2}"
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"

    img1 = np.array(Image.open(args.t1).convert("RGB"))
    img2 = np.array(Image.open(args.t2).convert("RGB"))
    assert img1.shape == img2.shape, f"Images must have matching dimensions. Got {img1.shape} vs {img2.shape}"

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = ChangePredictor(checkpoint_path=args.checkpoint, threshold=args.threshold)
    print(f"Running prediction on {img1.shape[1]}x{img1.shape[0]} scene...")

    res = predictor.predict_scene(
        t1_scene=img1,
        t2_scene=img2,
        patch_size=args.patch_size,
        overlap=args.overlap
    )

    # Save output mask
    os.makedirs(os.path.dirname(args.output_mask) or ".", exist_ok=True)
    mask_u8 = (res["binary_mask"] * 255).astype(np.uint8)
    Image.fromarray(mask_u8).save(args.output_mask)
    print(f"Saved binary change mask to: {args.output_mask}")

    if args.output_prob:
        os.makedirs(os.path.dirname(args.output_prob) or ".", exist_ok=True)
        prob_u8 = (res["probability_map"] * 255).astype(np.uint8)
        Image.fromarray(prob_u8).save(args.output_prob)
        print(f"Saved probability map to: {args.output_prob}")

    print("\n--- Prediction Summary ---")
    print(f"Changed Pixels:   {res['changed_pixels']:,} / {res['total_pixels']:,}")
    print(f"Change Magnitude: {res['change_percentage']}% of scene area")
    print(f"Applied Threshold:{res['threshold']}")

if __name__ == "__main__":
    main()
