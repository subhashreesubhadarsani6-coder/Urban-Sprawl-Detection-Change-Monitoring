"""Siamese U-Net Change Detection Model (FC-Siam-diff architecture).

Implements the fully convolutional Siamese difference network for bi-temporal
satellite change detection as described in Daudt et al.
Accepts two co-registered images (T1, T2) and outputs raw 1-channel spatial logits [B, 1, H, W].
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
    Handles odd/unaligned spatial dimensions via bilinear interpolation or reflection padding.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd dimensions if spatial sizes differ by 1 or more pixels
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
        """Pass one temporal observation through the shared encoder hierarchy."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        return [e1, e2, e3, e4, b]

    def forward(self, image_t1: torch.Tensor, image_t2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_t1: Earlier observation tensor [B, C, H, W]
            image_t2: Later observation tensor [B, C, H, W]

        Returns:
            Raw logit map tensor of shape [B, 1, H, W]
        """
        assert image_t1.shape == image_t2.shape, (
            f"T1 shape {image_t1.shape} does not match T2 shape {image_t2.shape}"
        )
        assert image_t1.shape[1] == self.in_channels, (
            f"Expected {self.in_channels} channels, got {image_t1.shape[1]}"
        )

        # 1. Extract shared multiscale features
        feats_t1 = self.forward_encoder(image_t1)
        feats_t2 = self.forward_encoder(image_t2)

        # 2. Compute multiscale absolute feature differences
        diff1 = torch.abs(feats_t1[0] - feats_t2[0])
        diff2 = torch.abs(feats_t1[1] - feats_t2[1])
        diff3 = torch.abs(feats_t1[2] - feats_t2[2])
        diff4 = torch.abs(feats_t1[3] - feats_t2[3])
        diff_bottle = torch.abs(feats_t1[4] - feats_t2[4])

        # 3. Decode through skip fusion
        d4 = self.dec4(diff_bottle, diff4)
        d3 = self.dec3(d4, diff3)
        d2 = self.dec2(d3, diff2)
        d1 = self.dec1(d2, diff1)

        # 4. Final raw logit output [B, 1, H, W]
        logits = self.out_conv(d1)
        return logits


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_millions": round(total / 1e6, 3),
        "trainable_millions": round(trainable / 1e6, 3)
    }


def get_model_summary(model: FCSiamDiff, input_size: Tuple[int, int] = (256, 256)) -> str:
    """Generate a readable textual summary of the model architecture."""
    params = count_parameters(model)
    return (
        f"========================================================\n"
        f"Architecture: FC-Siam-diff (Siamese U-Net)\n"
        f"Input Channels: {model.in_channels} (Bi-temporal T1 & T2)\n"
        f"Base Channels:  {model.base_channels}\n"
        f"Output Channels:{model.num_classes} (Binary Change Logits)\n"
        f"Total Params:   {params['total_parameters']:,} ({params['total_millions']}M)\n"
        f"Trainable:      {params['trainable_parameters']:,} ({params['trainable_millions']}M)\n"
        f"Input Shape:    2x [B, {model.in_channels}, {input_size[0]}, {input_size[1]}]\n"
        f"Output Shape:   [B, 1, {input_size[0]}, {input_size[1]}]\n"
        f"========================================================"
    )
