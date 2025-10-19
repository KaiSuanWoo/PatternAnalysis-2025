import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3×3×3 convolutions with InstanceNorm and ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


def down_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Max-pool followed by a conv block."""
    return nn.Sequential(
        nn.MaxPool3d(2),
        conv_block(in_ch, out_ch)
    )


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Upsample by factor 2 with transposed conv."""
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Baseline 3D U-Net with InstanceNorm skip connections."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4 = up_block(base_ch * 16, base_ch * 8)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)
        self.up3 = up_block(base_ch * 8, base_ch * 4)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = up_block(base_ch * 4, base_ch * 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = up_block(base_ch * 2, base_ch)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    @staticmethod
    def _align_skip(skip: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        """Resize skip tensor if needed to match decoder spatial dims."""
        if skip.shape[2:] != upsampled.shape[2:]:
            return F.interpolate(skip, size=upsampled.shape[2:], mode="trilinear", align_corners=False)
        return skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self._align_skip(e4, d4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._align_skip(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._align_skip(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._align_skip(e1, d1)], dim=1))

        out = self.out_conv(d1)
        return torch.sigmoid(out)

class AttentionGate3D(nn.Module):
    """
    3D Attention Gate (from Oktay et al., Attention U-Net 2018).
    Filters encoder features before skip concatenation.
    """
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_ch, inter_ch, 1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_ch, inter_ch, 1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ImprovedUNet3D(nn.Module):
    """U-Net + Attention Gates"""
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = down_block(base_ch, base_ch * 2)
        self.enc3 = down_block(base_ch * 2, base_ch * 4)
        self.enc4 = down_block(base_ch * 4, base_ch * 8)
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        # Attention gates
        self.ag4 = AttentionGate3D(g_ch=base_ch * 8, x_ch=base_ch * 8, inter_ch=base_ch * 4)
        self.ag3 = AttentionGate3D(g_ch=base_ch * 4, x_ch=base_ch * 4, inter_ch=base_ch * 2)
        self.ag2 = AttentionGate3D(g_ch=base_ch * 2, x_ch=base_ch * 2, inter_ch=base_ch)

        self.up4 = up_block(base_ch * 16, base_ch * 8)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)
        self.up3 = up_block(base_ch * 8, base_ch * 4)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = up_block(base_ch * 4, base_ch * 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = up_block(base_ch * 2, base_ch)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.up4(b)
        att4 = self.ag4(g=d4, x=self._align_skip(e4, d4))
        d4 = self.dec4(torch.cat([d4, att4], dim=1))

        d3 = self.up3(d4)
        att3 = self.ag3(g=d3, x=self._align_skip(e3, d3))
        d3 = self.dec3(torch.cat([d3, att3], dim=1))

        d2 = self.up2(d3)
        att2 = self.ag2(g=d2, x=self._align_skip(e2, d2))
        d2 = self.dec2(torch.cat([d2, att2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._align_skip(e1, d1)], dim=1))
        out = self.out_conv(d1)
        return torch.sigmoid(out)

    @staticmethod
    def _align_skip(skip: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != upsampled.shape[2:]:
            return F.interpolate(skip, size=upsampled.shape[2:], mode="trilinear", align_corners=False)
        return skip
