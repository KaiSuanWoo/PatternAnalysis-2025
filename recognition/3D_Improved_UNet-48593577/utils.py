import torch
import numpy as np
import random

# ================================================================
# ⚙️ SYSTEM HELPERS
# ================================================================

def check_cuda():
    """Return active torch.device and print GPU/CPU info."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
        return torch.device("cuda")
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device("cpu")


def set_seed(seed: int = 1337):
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}")

# ================================================================
# 🎯 METRICS
# ================================================================

def dice_coefficient(pred, target, eps=1e-6):
    """Computes Dice score for binary masks."""
    if target.dim() == 4:
        target = target.unsqueeze(1)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)

# ================================================================
# 🔧 LOSS DEFINITIONS
# ================================================================

class DiceLoss(torch.nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if target.dim() == 4:
            target = target.unsqueeze(1)
        inter = (pred * target).sum()
        denom = pred.sum() + target.sum()
        return 1 - (2 * inter + self.smooth) / (denom + self.smooth)


class FocalLoss(torch.nn.Module):
    """Binary Focal Loss (Lin et al. 2017)."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        bce = -(
            self.alpha * target * torch.log(pred)
            + (1 - self.alpha) * (1 - target) * torch.log(1 - pred)
        )
        pt = target * pred + (1 - target) * (1 - pred)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()


class HybridLoss(torch.nn.Module):
    """Combine Dice and Focal losses."""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, pred, target):
        return self.dw * self.dice(pred, target) + self.fw * self.focal(pred, target)
