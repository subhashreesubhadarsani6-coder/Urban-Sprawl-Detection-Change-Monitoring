"""Training Script for Siamese U-Net (FC-Siam-diff).

Supports:
- --smoke-test: Validates a single batch forward and backward pass.
- --overfit-test: Verifies pipeline ability to overfit a tiny subset.
- Full training with validation early stopping and checkpointing.
"""

import os
import sys
import argparse
import json
import yaml
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath("."))

from src.utils.reproducibility import set_seed, get_environment_info, seed_worker
from src.data.dataset import ChangeDetectionDataset
from src.data.splits import create_geographic_splits
from src.models.siamese_unet import FCSiamDiff, count_parameters, get_model_summary
from src.training.trainer import ChangeDetectionTrainer
from src.losses import CombinedBceDiceLoss

def run_smoke_test(model, dataset, device):
    """Execute a single-batch forward and backward pass to verify pipeline integrity."""
    print("\n--- Running 1-Batch Smoke Test ---")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    t1 = batch["image_t1"].to(device)
    t2 = batch["image_t2"].to(device)
    targets = batch["mask"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    criterion = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5)
    model.train()
    logits = model(t1, t2)
    loss = criterion(logits, targets, valid_mask=valid_mask)
    loss.backward()

    assert torch.isfinite(loss), f"Non-finite loss encountered in smoke test: {loss.item()}"
    assert logits.shape == (2, 1, 256, 256), f"Unexpected logit shape: {logits.shape}"
    print(f"Smoke test PASSED! Loss = {loss.item():.4f}, Output shape = {list(logits.shape)}")


def run_overfit_test(model, dataset, device, epochs=20):
    """Overfit on 2 patches to verify architecture learning capacity."""
    print(f"\n--- Running Overfit Test ({epochs} epochs on 2 patches) ---")
    subset = Subset(dataset, [0, 1])
    loader = DataLoader(subset, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = CombinedBceDiceLoss(bce_weight=0.5, dice_weight=0.5)

    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(1, epochs + 1):
        for batch in loader:
            t1 = batch["image_t1"].to(device)
            t2 = batch["image_t2"].to(device)
            targets = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = criterion(logits, targets, valid_mask=valid_mask)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}: Loss = {final_loss:.4f}")

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    print(f"Overfit test PASSED! Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train FC-Siam-diff change detection model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML configuration")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch smoke test and exit")
    parser.add_argument("--overfit-test", action="store_true", help="Run tiny overfit test and exit")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Reproducibility & Environment
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    env_info = get_environment_info()
    print("Environment Info:", json.dumps(env_info, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")

    # 2. Splits & Datasets
    splits_file = cfg["dataset"].get("splits_file", "configs/geographic_splits.json")
    splits = create_geographic_splits(output_path=splits_file, seed=seed)

    train_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=splits["train"],
        is_train_split=True,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=cfg["augmentation"]["enabled"]
    )

    val_ds = ChangeDetectionDataset(
        parquet_path=cfg["dataset"]["train_parquet"],
        split_scenes=splits["val"],
        is_train_split=True,
        patch_size=cfg["dataset"]["patch_size"],
        stride=cfg["dataset"]["stride"],
        augment=False
    )

    print(f"Dataset Loaded: {len(train_ds)} train patches, {len(val_ds)} validation patches")

    # 3. Model
    model = FCSiamDiff(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    print(get_model_summary(model))

    # Check for smoke test / overfit test
    if args.smoke_test:
        run_smoke_test(model, train_ds, device)
        return

    if args.overfit_test:
        run_overfit_test(model, train_ds, device)
        return

    # 4. DataLoaders
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 5. Trainer
    trainer = ChangeDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )

    print("\n--- Starting Full Model Training ---")
    results = trainer.fit()

    os.makedirs("reports", exist_ok=True)
    with open("reports/training_history.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Training Finished Successfully ---")
    print(f"Best Val F1: {results['best_f1']:.4f} (at threshold {results['best_threshold']})")
    print(f"Best Checkpoint: {results['best_checkpoint']}")

if __name__ == "__main__":
    main()
