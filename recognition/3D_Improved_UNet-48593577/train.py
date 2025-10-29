# train.py
import os, math, random, contextlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import find_pairs, split_by_patient, Prostate3DDataset, NUM_CLASSES
from modules import UNet3D_Improved

# ===============================================================
# CONFIGURATION (edit these as needed)
# ===============================================================
EPOCHS        = 200
BATCH_SIZE    = 2
ACCUM_STEPS   = 2                # gradient accumulation steps
PATCH_SIZE    = (128, 128, 96)
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
SEED          = 1337
SAVE_DIR      = "runs"
USE_DEEP_SUP  = True             # False disables deep supervision
WARMUP_STEPS  = 1000
TOTAL_STEPS_OVERRIDE = 0         # 0 = auto-calc from epochs
GRAD_CLIP_NORM = 12.0
# ===============================================================

# ----------------- Utils -----------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda"); device_type = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps"); device_type = "mps"
    else:
        device = torch.device("cpu"); device_type = "cpu"
    print(f"Using device: {device}")
    return device, device_type

def dice_per_class(probs, target, num_classes=NUM_CLASSES, eps=1e-6):
    """
    probs: (N,C,D,H,W) softmax probabilities
    target: (N,D,H,W) int64
    returns tensor of size (C-1,) for classes 1..C-1 (background ignored)
    """
    t_one = F.one_hot(target.long(), num_classes).permute(0,4,1,2,3).float()
    dices = []
    for c in range(1, num_classes):
        pc, tc = probs[:, c], t_one[:, c]
        inter = (pc * tc).sum(dim=(1,2,3))
        denom = pc.sum(dim=(1,2,3)) + tc.sum(dim=(1,2,3))
        dices.append(((2*inter + eps) / (denom + eps)).mean())
    return torch.stack(dices)  # (C-1,)

class DiceCELoss(torch.nn.Module):
    """ Combined Dice (ignore bg) + CrossEntropy with optional deep supervision. """
    def __init__(self, num_classes=NUM_CLASSES, ce_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weights = ce_weights

    def forward(self, logits, target):
        if isinstance(logits, list):  # deep supervision
            weights = [0.6, 0.2, 0.15, 0.05]
            return sum(w * self._single(l, target) for w, l in zip(weights, logits))
        return self._single(logits, target)

    def _single(self, logits, target):
        ce = F.cross_entropy(logits, target.long(), weight=self.ce_weights)
        probs = F.softmax(logits, dim=1)
        dice = 1.0 - dice_per_class(probs, target, self.num_classes).mean()
        return 0.5 * ce + 0.5 * dice

def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------- Training & Eval -----------------
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, device_type, grad_accum=1, log_every=50):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for it, (img, lab) in enumerate(loader, 1):
        img = img.to(device, non_blocking=(device_type=="cuda"))
        lab = lab.to(device, non_blocking=(device_type=="cuda"))

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device_type == "cuda" else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(img)
            loss = loss_fn(logits, lab) / grad_accum

        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            if it % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if it % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * grad_accum
        if log_every and it % log_every == 0:
            print(f"  iter {it:04d}/{len(loader)}  loss {running/it:.4f}")

    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, device_type):
    model.eval()
    dices = []
    for img, lab in loader:
        img = img.to(device, non_blocking=(device_type=="cuda"))
        lab = lab.to(device, non_blocking=(device_type=="cuda"))
        logits = model(img)
        if isinstance(logits, list): logits = logits[0]
        probs = F.softmax(logits, dim=1)
        d = dice_per_class(probs, lab)  # (C-1,)
        dices.append(d.unsqueeze(0))
    if not dices:
        return torch.zeros(NUM_CLASSES-1)
    return torch.cat(dices, 0).mean(0).cpu()

# ----------------- Main -----------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)

    device, device_type = setup_device()

    # ---------- Data ----------
    pairs = find_pairs()
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, train_ratio=0.7, val_ratio=0.15, seed=SEED)
    print(f"Split → train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

    train_ds = Prostate3DDataset(train_pairs, patch_size=PATCH_SIZE, augment=True,  patch_prob_fg=0.6)
    val_ds   = Prostate3DDataset(val_pairs,   patch_size=PATCH_SIZE, augment=False, patch_prob_fg=0.0)

    pin = (device_type == "cuda")
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=max(1, NUM_WORKERS//2), pin_memory=pin)

    # ---------- Model ----------
    model = UNet3D_Improved(
        in_channels=1,
        num_classes=NUM_CLASSES,
        deep_supervision=USE_DEEP_SUP,
        base=32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # GradScaler on CUDA only
    scaler = torch.amp.GradScaler() if device_type == "cuda" else None

    loss_fn = DiceCELoss(NUM_CLASSES)

    total_steps = TOTAL_STEPS_OVERRIDE or (EPOCHS * max(1, len(train_ld)//max(1, ACCUM_STEPS)))
    scheduler = make_scheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps)

    # ---------- Train ----------
    best_mean = -1.0
    best_path = os.path.join(SAVE_DIR, "best_improved_unet3d.pt")
    step = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model, train_ld, optimizer, scaler, loss_fn,
            device, device_type, grad_accum=ACCUM_STEPS
        )

        # step-wise scheduler: advance by number of optimizer steps taken this epoch
        steps_this_epoch = math.ceil(len(train_ld) / max(1, ACCUM_STEPS))
        for _ in range(steps_this_epoch):
            scheduler.step(); step += 1

        val_dice = evaluate(model, val_ld, device, device_type)   # tensor (C-1,)
        mean_dice = float(val_dice.mean().item())
        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | val mean Dice {mean_dice:.3f} | per-class {val_dice.tolist()}")

        if mean_dice > best_mean:
            best_mean = mean_dice
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best mean Dice {best_mean:.3f} — saved to {best_path}")

    print(f"Training complete. Best val mean Dice: {best_mean:.3f}")

if __name__ == "__main__":
    main()
