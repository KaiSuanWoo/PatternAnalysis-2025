import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Blocks ---

def conv_block(in_ch, out_ch):
    """Two 3x3x3 convs + InstanceNorm + ReLU"""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )

def down_block(in_ch, out_ch):
    return nn.Sequential(nn.MaxPool3d(2), conv_block(in_ch, out_ch))

def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )

# --- Baseline 3D U-Net ---

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = down_block(base_ch, base_ch * 2)
        self.enc3 = down_block(base_ch * 2, base_ch * 4)
        self.enc4 = down_block(base_ch * 4, base_ch * 8)
        self.enc5 = down_block(base_ch * 8, base_ch * 16)

        self.bottleneck = conv_block(base_ch * 16, base_ch * 32)

        self.up5 = up_block(base_ch * 32, base_ch * 16)
        self.dec5 = conv_block(base_ch * 32, base_ch * 16)
        self.up4 = up_block(base_ch * 16, base_ch * 8)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)
        self.up3 = up_block(base_ch * 8, base_ch * 4)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = up_block(base_ch * 4, base_ch * 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = up_block(base_ch * 2, base_ch)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, 1)

    @staticmethod
    def _match_size(src, ref):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return src

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)

        d5 = self.up5(b)
        skip5 = self._match_size(e5, d5)
        d5 = self.dec5(torch.cat([d5, skip5], dim=1))
        d4 = self.up4(d5)
        skip4 = self._match_size(e4, d4)
        d4 = self.dec4(torch.cat([d4, skip4], dim=1))
        d3 = self.up3(d4)
        skip3 = self._match_size(e3, d3)
        d3 = self.dec3(torch.cat([d3, skip3], dim=1))
        d2 = self.up2(d3)
        skip2 = self._match_size(e2, d2)
        d2 = self.dec2(torch.cat([d2, skip2], dim=1))
        d1 = self.up1(d2)
        skip1 = self._match_size(e1, d1)
        d1 = self.dec1(torch.cat([d1, skip1], dim=1))
        out = self.out_conv(d1)
        return torch.sigmoid(out)
