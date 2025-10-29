import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Building blocks
# -------------------------
def conv3x3(in_ch, out_ch, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation to reweight channels (lightweight)."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(ch, ch // r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch // r, ch, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class ResidualConvBlock3D(nn.Module):
    """
    Two 3x3 convs with GroupNorm + ReLU and a residual connection.
    Optionally applies SE channel attention at the end.
    """
    def __init__(self, in_ch, out_ch, groups=8, se=True, dropout=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.se    = SEBlock3D(out_ch) if se else nn.Identity()
        self.drop  = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        self.proj  = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.relu(self.gn2(self.conv2(out))))
        out = self.se(out)
        out = self.relu(out + identity)
        return out


class AttGate3D(nn.Module):
    """
    Attention gate from "Attention U-Net".
    Takes encoder feature (skip) and decoder gate (g).
    """
    def __init__(self, in_ch_skip, in_ch_g, inter_ch):
        super().__init__()
        self.theta = conv1x1(in_ch_skip, inter_ch)
        self.phi   = conv1x1(in_ch_g, inter_ch)
        self.psi   = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(inter_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x_skip, g):
        # match spatial dims by interpolation of g
        if g.shape[-3:] != x_skip.shape[-3:]:
            g = F.interpolate(g, size=x_skip.shape[-3:], mode="trilinear", align_corners=False)
        att = self.theta(x_skip) + self.phi(g)
        att = self.psi(att)
        return x_skip * att


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_att=False, skip_ch=None):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.use_att = use_att
        if use_att:
            assert skip_ch is not None, "skip_ch must be provided when use_att=True"
            self.att = AttGate3D(in_ch_skip=skip_ch, in_ch_g=out_ch, inter_ch=max(out_ch // 2, 1))
            self.fuse = ResidualConvBlock3D(out_ch + skip_ch, out_ch)
        else:
            self.fuse = ResidualConvBlock3D(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # center-crop/pad skip if small mismatch
        if x.shape[-3:] != skip.shape[-3:]:
            skip = F.interpolate(skip, size=x.shape[-3:], mode="trilinear", align_corners=False)

        if self.use_att:
            skip = self.att(skip, x)
            x = torch.cat([x, skip], dim=1)
        else:
            x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


# -------------------------
# UNet backbone
# -------------------------
class UNet3D(nn.Module):
    """
    Improved 3D U-Net:
      - Residual conv blocks + GroupNorm
      - Optional SE channel attention inside blocks
      - Optional Attention Gates on skip connections
    """
    def __init__(self, in_channels=1, num_classes=2, base_ch=32, depth=4,
                 use_se=True, use_att=True, dropout=0.0):
        super().__init__()
        assert depth in (3, 4, 5), "Depth 3-5 is typical for 3D with memory limits."

        chs = [base_ch * (2 ** i) for i in range(depth)]  # encoder widths

        # Encoder
        self.enc0 = ResidualConvBlock3D(in_channels, chs[0], se=use_se, dropout=dropout)
        self.down0 = nn.MaxPool3d(2)

        self.enc1 = ResidualConvBlock3D(chs[0], chs[1], se=use_se, dropout=dropout)
        self.down1 = nn.MaxPool3d(2)

        if depth >= 4:
            self.enc2 = ResidualConvBlock3D(chs[1], chs[2], se=use_se, dropout=dropout)
            self.down2 = nn.MaxPool3d(2)

        if depth == 5:
            self.enc3 = ResidualConvBlock3D(chs[2], chs[3], se=use_se, dropout=dropout)
            self.down3 = nn.MaxPool3d(2)

        # Bottleneck (match encoder stage just before the deepest downsample)
        if depth == 5:
            bott_in = chs[3]   # enc3 out
        elif depth == 4:
            bott_in = chs[2]   # enc2 out
        else:  # depth == 3
            bott_in = chs[1]   # enc1 out

        self.bott = ResidualConvBlock3D(bott_in, bott_in * 2, se=use_se, dropout=dropout)

        # Decoder
        if depth == 5:
            self.up3 = UpBlock3D(in_ch=bott_in * 2, out_ch=chs[3], use_att=use_att, skip_ch=chs[3])
            self.up2 = UpBlock3D(in_ch=chs[3],       out_ch=chs[2], use_att=use_att, skip_ch=chs[2])
            self.up1 = UpBlock3D(in_ch=chs[2],       out_ch=chs[1], use_att=use_att, skip_ch=chs[1])
            self.up0 = UpBlock3D(in_ch=chs[1],       out_ch=chs[0], use_att=use_att, skip_ch=chs[0])
            dec_out = chs[0]
        elif depth == 4:
            self.up2 = UpBlock3D(in_ch=bott_in * 2, out_ch=chs[2], use_att=use_att, skip_ch=chs[2])
            self.up1 = UpBlock3D(in_ch=chs[2],       out_ch=chs[1], use_att=use_att, skip_ch=chs[1])
            self.up0 = UpBlock3D(in_ch=chs[1],       out_ch=chs[0], use_att=use_att, skip_ch=chs[0])
            dec_out = chs[0]
        else:  # depth == 3
            self.up1 = UpBlock3D(in_ch=bott_in * 2, out_ch=chs[1], use_att=use_att, skip_ch=chs[1])
            self.up0 = UpBlock3D(in_ch=chs[1],       out_ch=chs[0], use_att=use_att, skip_ch=chs[0])
            dec_out = chs[0]


        self.head = nn.Conv3d(dec_out, num_classes, kernel_size=1)

        self.depth = depth

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        x = self.down0(e0)

        e1 = self.enc1(x)
        if self.depth >= 4:
            x = self.down1(e1)
            e2 = self.enc2(x)

        if self.depth == 5:
            x = self.down2(e2)
            e3 = self.enc3(x)
            x = self.down3(e3)
            # Bottleneck
            x = self.bott(x)
            # Decoder
            x = self.up3(x, e3)
            x = self.up2(x, e2)
            x = self.up1(x, e1)
            x = self.up0(x, e0)
        elif self.depth == 4:
            x = self.down2(e2)
            x = self.bott(x)
            x = self.up2(x, e2)
            x = self.up1(x, e1)
            x = self.up0(x, e0)
        else:  # depth == 3
            x = self.down1(e1)
            x = self.bott(x)
            x = self.up1(x, e1)
            x = self.up0(x, e0)

        logits = self.head(x)
        return logits
