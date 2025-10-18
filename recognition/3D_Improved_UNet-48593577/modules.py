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
