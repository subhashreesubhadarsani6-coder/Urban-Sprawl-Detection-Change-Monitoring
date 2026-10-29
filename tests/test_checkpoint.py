"""Unit tests for Checkpoint Management Module."""

import os
import pytest
import torch
from src.models.siamese_unet import FCSiamDiff
from src.training.checkpoint import save_checkpoint, load_checkpoint

def test_checkpoint_save_and_reload(tmp_path):
    ckpt_path = str(tmp_path / "test_model.pth")
    model = FCSiamDiff(in_channels=3, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Put in evaluation mode and record output on fixed tensor
    model.eval()
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        orig_out = model(t1, t2)

    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        epoch=5,
        best_metric=0.825,
        config={"test": True},
        threshold=0.45
    )

    assert os.path.exists(ckpt_path)

    # Create new model instance and reload
    new_model = FCSiamDiff(in_channels=3, base_channels=16)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    ckpt_meta = load_checkpoint(ckpt_path, new_model, new_opt)

    assert ckpt_meta["epoch"] == 5
    assert ckpt_meta["best_metric"] == 0.825
    assert ckpt_meta["threshold"] == 0.45

    new_model.eval()
    with torch.no_grad():
        restored_out = new_model(t1, t2)

    assert torch.allclose(orig_out, restored_out, atol=1e-6), "Reloaded model weights produced differing output!"

def test_missing_checkpoint_raises_error():
    model = FCSiamDiff(in_channels=3, base_channels=16)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_ckpt_123.pth", model)

def test_incompatible_checkpoint_raises_error(tmp_path):
    ckpt_path = str(tmp_path / "incompatible.pth")
    # Save a 4-channel model
    model4 = FCSiamDiff(in_channels=4, base_channels=16)
    opt4 = torch.optim.AdamW(model4.parameters(), lr=1e-3)
    save_checkpoint(ckpt_path, model4, opt4, epoch=1, best_metric=0.5, config={})

    # Try to load into a 3-channel model
    model3 = FCSiamDiff(in_channels=3, base_channels=16)
    with pytest.raises(ValueError, match="Incompatible channel count"):
        load_checkpoint(ckpt_path, model3)
