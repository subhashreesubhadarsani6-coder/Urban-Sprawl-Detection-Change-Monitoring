"""Register and verify trained model checkpoint for production inference."""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath("."))

from src.models.siamese_unet import FCSiamDiff, count_parameters
from src.training.checkpoint import load_checkpoint

def register_checkpoint(
    checkpoint_path: str = "checkpoints/best_checkpoint.pth",
    registry_file: str = "checkpoints/registry.json"
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    in_channels = ckpt.get("in_channels", 3)
    base_channels = ckpt.get("base_channels", 32)
    num_classes = ckpt.get("num_classes", 1)

    # Verify model reload
    model = FCSiamDiff(in_channels=in_channels, base_channels=base_channels, num_classes=num_classes)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    params = count_parameters(model)
    file_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)

    entry = {
        "status": "REGISTERED",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "file_size_mb": file_size_mb,
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "threshold": ckpt.get("threshold", 0.5),
        "architecture": "FC-Siam-diff",
        "in_channels": in_channels,
        "base_channels": base_channels,
        "parameters": params["total_parameters"],
        "parameters_m": params["total_millions"]
    }

    os.makedirs(os.path.dirname(registry_file) or ".", exist_ok=True)
    with open(registry_file, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"Checkpoint verified and registered successfully: {registry_file}")
    print(json.dumps(entry, indent=2))
    return entry

def main():
    parser = argparse.ArgumentParser(description="Register a trained model checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pth")
    parser.add_argument("--registry", default="checkpoints/registry.json")
    args = parser.parse_args()

    register_checkpoint(args.checkpoint, args.registry)

if __name__ == "__main__":
    main()
