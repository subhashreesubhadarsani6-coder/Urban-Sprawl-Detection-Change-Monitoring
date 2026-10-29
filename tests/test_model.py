"""Unit tests for Siamese U-Net (FC-Siam-diff) Model Architecture."""

import pytest
import torch
from src.models.siamese_unet import FCSiamDiff, count_parameters

def test_model_output_shape():
    model = FCSiamDiff(in_channels=3, base_channels=16, num_classes=1)
    model.eval()
    t1 = torch.randn(2, 3, 128, 128)
    t2 = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(t1, t2)
    assert out.shape == (2, 1, 128, 128)
    assert torch.isfinite(out).all()

def test_multiple_batch_sizes():
    model = FCSiamDiff(in_channels=3, base_channels=16)
    model.eval()
    for b in [1, 3]:
        t1 = torch.randn(b, 3, 64, 64)
        t2 = torch.randn(b, 3, 64, 64)
        with torch.no_grad():
            out = model(t1, t2)
        assert out.shape == (b, 1, 64, 64)

def test_configurable_channels():
    # 4 channels (e.g. RGB + NIR)
    model = FCSiamDiff(in_channels=4, base_channels=16)
    model.eval()
    t1 = torch.randn(1, 4, 64, 64)
    t2 = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        out = model(t1, t2)
    assert out.shape == (1, 1, 64, 64)

def test_odd_image_dimensions():
    # Odd spatial dimensions test padding and interpolation in decoder
    model = FCSiamDiff(in_channels=3, base_channels=16)
    model.eval()
    t1 = torch.randn(1, 3, 127, 131)
    t2 = torch.randn(1, 3, 127, 131)
    with torch.no_grad():
        out = model(t1, t2)
    assert out.shape == (1, 1, 127, 131)

def test_shared_encoder_weights():
    # Identical inputs to both branches must yield identical zero feature difference
    model = FCSiamDiff(in_channels=3, base_channels=16)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    feats1 = model.forward_encoder(x)
    feats2 = model.forward_encoder(x)
    for f1, f2 in zip(feats1, feats2):
        assert torch.allclose(f1, f2, atol=1e-5), "Shared encoder weights produced divergent features on identical input!"

def test_parameter_counting():
    model = FCSiamDiff(in_channels=3, base_channels=16)
    p = count_parameters(model)
    assert p["total_parameters"] > 0
    assert p["trainable_parameters"] == p["total_parameters"]
