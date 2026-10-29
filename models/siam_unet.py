"""Siamese U-Net Change Detection Model (FC-Siam-diff architecture).

Implements the fully convolutional Siamese difference network for bi-temporal
satellite change detection as described in Daudt et al.
Accepts two co-registered images (T1, T2) and outputs raw 1-channel spatial logits [B, 1, H, W].
Shared-weight encoder processes both temporal inputs and extracts multiscale features.
Absolute feature difference D_i = |F_T1_i - F_T2_i| is computed at each stage and
fused through a U-Net decoder with skip connections.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential [Conv2d -> BatchNorm2d -> ReLU] blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Upsampling block that fuses skip difference features with decoder feature maps.
    Handles odd/unaligned spatial dimensions via bilinear interpolation or padding.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FCSiamDiff(nn.Module):
    """FC-Siam-diff: Fully Convolutional Siamese Difference U-Net.

    Encodes T1 and T2 using a shared-weight multiscale encoder, computes
    the absolute feature difference |f_t1 - f_t2| at each scale, and
    fuses these differences through a U-Net decoder with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        c = base_channels

        # Shared-weight Encoder Stages
        self.enc1 = DoubleConv(in_channels, c)        # -> [B, c, H, W]
        self.pool1 = nn.MaxPool2d(2, 2)              # -> [B, c, H/2, W/2]

        self.enc2 = DoubleConv(c, c * 2)             # -> [B, 2c, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2, 2)              # -> [B, 2c, H/4, W/4]

        self.enc3 = DoubleConv(c * 2, c * 4)         # -> [B, 4c, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2, 2)              # -> [B, 4c, H/8, W/8]

        self.enc4 = DoubleConv(c * 4, c * 8)         # -> [B, 8c, H/8, W/8]
        self.pool4 = nn.MaxPool2d(2, 2)              # -> [B, 8c, H/16, W/16]

        # Bottleneck (bridge)
        self.bottleneck = DoubleConv(c * 8, c * 16)  # -> [B, 16c, H/16, W/16]

        # Decoder Stages (fusing absolute differences from encoder skips)
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Final 1x1 classification head outputting raw logits
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward_encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Pass one temporal observation through the shared encoder hierarchy.

        Returns:
            List of feature maps [e1, e2, e3, e4, bottleneck]
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        return [e1, e2, e3, e4, b]

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Siamese difference network.

        Args:
            t1: Earlier observation [B, C, H, W]
            t2: Later observation [B, C, H, W]

        Returns:
            Raw logits [B, 1, H, W] (No sigmoid applied, for BCEWithLogitsLoss).
        """
        # 1. Shared Encoder Feature Extraction
        feats_t1 = self.forward_encoder(t1)
        feats_t2 = self.forward_encoder(t2)

        e1_1, e2_1, e3_1, e4_1, b1 = feats_t1
        e1_2, e2_2, e3_2, e4_2, b2 = feats_t2

        # 2. Multiscale Absolute Feature Difference Calculation
        # D_i = |F_T1_i - F_T2_i|
        diff1 = torch.abs(e1_1 - e1_2)
        diff2 = torch.abs(e2_1 - e2_2)
        diff3 = torch.abs(e3_1 - e3_2)
        diff4 = torch.abs(e4_1 - e4_2)
        diff_b = torch.abs(b1 - b2)

        # 3. U-Net Decoder with Difference Skips
        d4 = self.dec4(diff_b, diff4)
        d3 = self.dec3(d4, diff3)
        d2 = self.dec2(d3, diff2)
        d1 = self.dec1(d2, diff1)

        # 4. Final Classification Head
        logits = self.out_conv(d1)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Generate architecture summary string."""
        params = self.count_parameters()
        return (
            f"FCSiamDiff(in_channels={self.in_channels}, "
            f"base_channels={self.base_channels}, "
            f"num_classes={self.num_classes}) | "
            f"Trainable Parameters: {params:,} ({params/1e6:.2f}M)"
        )


def test_forward():
    """Test model with dummy tensors as requested in Phase C."""
    model = FCSiamDiff(in_channels=3, base_channels=32, num_classes=1)
    t1 = torch.randn(2, 3, 256, 256)
    t2 = torch.randn(2, 3, 256, 256)
    out = model(t1, t2)
    assert out.shape == (2, 1, 256, 256), f"Unexpected output shape: {out.shape}"
    print(model.summary())
    print("Test forward pass successful! Output shape:", out.shape)


if __name__ == "__main__":
    test_forward()
