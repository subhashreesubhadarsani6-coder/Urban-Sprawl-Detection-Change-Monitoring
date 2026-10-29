"""Trainer Module for Siamese U-Net Change Detection.

Implements the complete training and validation loops with AdamW optimizer,
cosine learning rate scheduling, Automatic Mixed Precision (AMP), gradient clipping,
early stopping, and metric tracking.
"""

from typing import Dict, Any, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses import CombinedBceDiceLoss
from ..metrics import ChangeDetectionMetrics, compute_metrics_at_thresholds
from .checkpoint import save_checkpoint

class ChangeDetectionTrainer:
    """Manages model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = float(train_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.patience = train_cfg.get("patience", 8)
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.checkpoint_dir = config.get("checkpoints", {}).get("dir", "checkpoints")

        # Loss function with imbalance handling
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedBceDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            pos_weight=float(loss_cfg.get("pos_weight", 1.0)) if "pos_weight" in loss_cfg else None
        )

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # State tracking
        self.best_metric = -1.0
        self.best_threshold = 0.5
        self.epochs_without_improvement = 0
        self.history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch across all training batches."""
        self.model.train()
        total_loss = 0.0
        tracker = ChangeDetectionMetrics()
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            t1 = batch["image_t1"].to(self.device, non_blocking=True)
            t2 = batch["image_t2"].to(self.device, non_blocking=True)
            targets = batch["mask"].to(self.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(t1, t2)
                    loss = self.criterion(logits, targets, valid_mask=valid_mask)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(t1, t2)
                loss = self.criterion(logits, targets, valid_mask=valid_mask)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            tracker.update(logits, targets, valid_mask=valid_mask)

        duration = time.time() - start_time
        avg_loss = total_loss / max(1, len(self.train_loader))
        metrics = tracker.compute()
        metrics["loss"] = round(avg_loss, 5)
        metrics["duration_sec"] = round(duration, 2)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate model on independent validation scenes across candidate thresholds."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []
        all_valid = []

        with torch.no_grad():
            for batch in self.val_loader:
                t1 = batch["image_t1"].to(self.device, non_blocking=True)
                t2 = batch["image_t2"].to(self.device, non_blocking=True)
                targets = batch["mask"].to(self.device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(t1, t2)
                        loss = self.criterion(logits, targets, valid_mask=valid_mask)
                else:
                    logits = self.model(t1, t2)
                    loss = self.criterion(logits, targets, valid_mask=valid_mask)

                total_loss += loss.item()
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_valid.append(valid_mask.cpu().numpy())

        avg_loss = total_loss / max(1, len(self.val_loader))

        probs_np = np.concatenate(all_probs, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        valid_np = np.concatenate(all_valid, axis=0)

        # Threshold search over validation set only
        candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        thresh_metrics = compute_metrics_at_thresholds(
            probs_np, targets_np, candidate_thresholds, valid_mask=valid_np
        )

        # Select threshold maximizing validation F1 score
        best_t = 0.5
        best_f1 = -1.0
        for t, m in thresh_metrics.items():
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t = t

        val_metrics = dict(thresh_metrics[best_t])
        val_metrics["loss"] = round(avg_loss, 5)
        val_metrics["selected_threshold"] = best_t
        val_metrics["candidate_thresholds"] = {str(k): v for k, v in thresh_metrics.items()}
        return val_metrics

    def fit(self) -> Dict[str, Any]:
        """Run full training loop with validation, early stopping, and checkpointing."""
        best_ckpt_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
        latest_ckpt_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics
            }
            self.history.append(epoch_record)

            val_f1 = val_metrics["f1"]
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} F1: {val_f1:.4f} (Thresh: {val_metrics['selected_threshold']}) | "
                f"Time: {train_metrics['duration_sec']}s",
                flush=True
            )

            # Check for best model
            if val_f1 > self.best_metric:
                self.best_metric = val_f1
                self.best_threshold = val_metrics["selected_threshold"]
                self.epochs_without_improvement = 0
                save_checkpoint(
                    best_ckpt_path,
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    config=self.config,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    threshold=self.best_threshold
                )
                print(f"  --> Saved new best checkpoint (Val F1: {self.best_metric:.4f})", flush=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break

            # Save latest checkpoint
            save_checkpoint(
                latest_ckpt_path,
                self.model,
                self.optimizer,
                epoch=epoch,
                best_metric=self.best_metric,
                config=self.config,
                scheduler=self.scheduler,
                scaler=self.scaler,
                threshold=val_metrics["selected_threshold"]
            )

        return {
            "best_f1": self.best_metric,
            "best_threshold": self.best_threshold,
            "best_checkpoint": best_ckpt_path,
            "history": self.history
        }
