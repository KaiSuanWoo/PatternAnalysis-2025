# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- blocks ----
class Conv3dIN(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv3d(cin, cout, k, s, p, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv3dIN(c, c)
        self.conv2 = Conv3dIN(c, c)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class scSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv3d(c // r, c, 1), nn.Sigmoid()
        )
        self.sSE = nn.Sequential(nn.Conv3d(c, 1, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

# ---- encoder/decoder with MPS-friendly ops ----
class Down(nn.Module):
    """
    MPS-safe downsampling: stride-2 3D conv instead of MaxPool3d.
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.down = nn.Conv3d(cin, cin, kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
            Conv3dIN(cin, cout),
            ResBlock3D(cout),
            scSE(cout),
        )
    def forward(self, x):
        x = self.down(x)
        return self.conv(x)

class Up(nn.Module):
    """
    MPS-safe upsampling: trilinear upsample + 1x1 conv (no ConvTranspose3d).
    Note: after upsample we concat with skip (channels=cout), so input to conv = 2*cout.
    """
    def __init__(self, cin, cout):
        super().__init__()
        # reduce channels to cout after upsample
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce = nn.Conv3d(cin, cout, kernel_size=1, stride=1, bias=False)
        self.conv = nn.Sequential(
            Conv3dIN(cout * 2, cout),
            ResBlock3D(cout),
            scSE(cout),
        )

    def forward(self, x, skip):
        x = self.up(x)           # (N, cin, ..) -> spatial up by 2
        x = self.reduce(x)       # (N, cout, ..)
        # pad if needed for odd dims
        diff_d = skip.size(-3) - x.size(-3)
        diff_h = skip.size(-2) - x.size(-2)
        diff_w = skip.size(-1) - x.size(-1)
        if diff_d or diff_h or diff_w:
            x = F.pad(x, [0, max(0, diff_w), 0, max(0, diff_h), 0, max(0, diff_d)])
        x = torch.cat([skip, x], dim=1)  # channels = cout + cout = 2*cout
        return self.conv(x)

# ---- UNet3D Improved ----
class UNet3D_Improved(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base=32, deep_supervision=True):
        super().__init__()
        chs = [base, base*2, base*4, base*8, base*10]
        self.ds = deep_supervision

        self.in_conv = nn.Sequential(
            Conv3dIN(in_channels, chs[0]),
            ResBlock3D(chs[0]),
            scSE(chs[0]),
        )
        self.d1 = Down(chs[0], chs[1])
        self.d2 = Down(chs[1], chs[2])
        self.d3 = Down(chs[2], chs[3])

        self.bn = nn.Sequential(
            Conv3dIN(chs[3], chs[4]),
            ResBlock3D(chs[4]),
            nn.Dropout3d(0.1),
        )

        self.u3 = Up(chs[4], chs[3])  # up: 10b -> 8b
        self.u2 = Up(chs[3], chs[2])  # up: 8b -> 4b
        self.u1 = Up(chs[2], chs[1])  # up: 4b -> 2b
        self.u0 = Up(chs[1], chs[0])  # up: 2b -> 1b

        self.out0 = nn.Conv3d(chs[0], num_classes, 1)
        if self.ds:
            self.out1 = nn.Conv3d(chs[1], num_classes, 1)
            self.out2 = nn.Conv3d(chs[2], num_classes, 1)
            self.out3 = nn.Conv3d(chs[3], num_classes, 1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        xb = self.bn(x3)

        y3 = self.u3(xb, x3)
        y2 = self.u2(y3, x2)
        y1 = self.u1(y2, x1)
        y0 = self.u0(y1, x0)

        logits0 = self.out0(y0)
        if not self.ds:
            return logits0

        # deep supervision: upsample aux heads to full res
        logits1 = F.interpolate(self.out1(y1), size=logits0.shape[-3:], mode='trilinear', align_corners=False)
        logits2 = F.interpolate(self.out2(y2), size=logits0.shape[-3:], mode='trilinear', align_corners=False)
        logits3 = F.interpolate(self.out3(y3), size=logits0.shape[-3:], mode='trilinear', align_corners=False)
        return [logits0, logits1, logits2, logits3]
