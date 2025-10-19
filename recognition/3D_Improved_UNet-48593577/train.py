import os, csv, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import find_patients, ProstatePatchDataset
from modules import UNet3D, ImprovedUNet3D
from utils import (
    check_cuda, set_seed, dice_coefficient,
    DiceLoss, HybridLoss
)

# -------------------- CONFIG (edit here) --------------------
CFG = {
    "DATA_ROOT": "Prostate3D_data",
    "SAVE_DIR":  "runs",

    "MODEL": "improved",          # "unet" or "improved"
    "LOSS":  "dice",              # "dice" or "hybrid"

    "EPOCHS": 20,
    "BATCH_SIZE": 1,
    "PATCH": (64, 64, 64),        # safe on CPU; use (96,96,96) on GPU
    "SEED": 1337,
    "SUBSET": 40,                 # 0 = all; small number for quick tests

    "NUM_WORKERS": 0,
    "PIN_MEMORY": False,

    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-4,
    "GRAD_CLIP": 1.0,

    "LR_PATIENCE": 5,
    "LR_FACTOR": 0.5,
    "EARLY_STOP_TARGET": 0.70,    # stop once val Dice ≥ target for N epochs
    "EARLY_STOP_PATIENCE": 2,

    "USE_AMP": torch.cuda.is_available(),

    # Hybrid loss focal alpha: if None, auto-estimate from foreground frequency
    "FOCAL_ALPHA": None,
    "FOCAL_GAMMA": 2.0,
}
# -------------------------------------------------------------

def _match_spatial_size(t: torch.Tensor, ref: torch.Tensor, is_mask: bool) -> torch.Tensor:
    """Resize tensor t to ref's (D,H,W)."""
    if t.shape[-3:] == ref.shape[-3:]:
        return t
    size = ref.shape[-3:]
    if is_mask:
        return F.interpolate(t, size=size, mode="nearest")
    else:
        return F.interpolate(t, size=size, mode="trilinear", align_corners=False)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

@torch.no_grad()
def _estimate_fg_alpha(loader, max_batches=8):
    """
    Estimate Focal alpha from foreground prevalence.
    alpha ~ (1 - fg_fraction) so minority class gets higher weight.
    """
    total_vox = 0
    fg_vox = 0
    seen = 0
    for batch in loader:
        y = batch["mask"]
        total_vox += y.numel()
        fg_vox += (y > 0).sum().item()
        seen += 1
        if seen >= max_batches: break
    fg_frac = max(1e-6, min(1.0 - 1e-6, fg_vox / max(1, total_vox)))
    alpha = 1.0 - fg_frac
    return float(alpha), float(fg_frac)

def _overlay_central_axial(img_chn_first: torch.Tensor,  # (1,D,H,W)
                           mask_bin: torch.Tensor,       # (1,D,H,W) or (D,H,W)
                           pred_bin: torch.Tensor,       # (1,D,H,W) or (D,H,W)
                           save_path: str):
    """
    Save a 3x1 grid: image, GT, Pred (central axial slice).
    """
    ensure_dir(os.path.dirname(save_path))
    if img_chn_first.dim() == 4:
        img = img_chn_first[0].cpu().numpy()  # (D,H,W)
    else:
        img = img_chn_first.cpu().numpy()
    if mask_bin.dim() == 4: mask = mask_bin[0].cpu().numpy()
    else:                   mask = mask_bin.cpu().numpy()
    if pred_bin.dim() == 4: pred = pred_bin[0].cpu().numpy()
    else:                   pred = pred_bin.cpu().numpy()

    D, H, W = img.shape
    z = D // 2
    img2d = img[z]
    gt2d  = mask[z]
    pr2d  = pred[z]

    # normalise image to 0..1 for display
    img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(img2d, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(gt2d, cmap="gray");  axs[1].set_title("GT");    axs[1].axis("off")
    axs[2].imshow(pr2d, cmap="gray");  axs[2].set_title("Pred");  axs[2].axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def train_one_epoch(model, loader, opt, loss_fn, device, use_amp=True, grad_clip=1.0):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)                        # (N,1,D,H,W)
        y = batch["mask"].float().to(device).unsqueeze(1)    # -> (N,1,D,H,W)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            p = model(x)
            y_ = _match_spatial_size(y, p, is_mask=True)
            p_ = _match_spatial_size(p, y_, is_mask=False)
            loss = loss_fn(p_, y_)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update()

        running += loss.item()
        pbar.set_postfix(loss=float(loss))
    return running / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device, use_amp=False):
    model.eval()
    dices = []
    pbar = tqdm(loader, desc="Valid", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["mask"].float().to(device).unsqueeze(1)
        with torch.amp.autocast("cuda", enabled=use_amp):
            p = model(x)
            y_ = _match_spatial_size(y, p, is_mask=True)
            p_ = _match_spatial_size(p, y_, is_mask=False)
            dices.append(dice_coefficient(p_, y_).item())
    return float(np.mean(dices)) if dices else 0.0

def main():
    cfg = CFG
    device = check_cuda()
    set_seed(cfg["SEED"])
    ensure_dir(cfg["SAVE_DIR"])

    # ---------- split ----------
    items = find_patients(cfg["DATA_ROOT"])
    ids = [it.pid for it in items]
    if cfg["SUBSET"] and cfg["SUBSET"] < len(ids):
        ids = ids[:cfg["SUBSET"]]
    n_train = int(0.7 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    print(f"Split → train: {len(train_ids)}, val: {len(val_ids)}")

    # ---------- data ----------
    patch = tuple(cfg["PATCH"])
    train_ds = ProstatePatchDataset(cfg["DATA_ROOT"], train_ids, patch_size=patch,
                                    mode="train", augment=True, normImage=True, orient=True)
    val_ds   = ProstatePatchDataset(cfg["DATA_ROOT"], val_ids,   patch_size=patch,
                                    mode="val", augment=False, normImage=True, orient=True)
    pin_memory = bool(cfg["PIN_MEMORY"] and torch.cuda.is_available())
    tr_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True,
                           num_workers=cfg["NUM_WORKERS"], pin_memory=pin_memory)
    va_loader = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                           num_workers=cfg["NUM_WORKERS"], pin_memory=pin_memory)

    # ---------- model ----------
    if cfg["MODEL"].lower() == "improved":
        model = ImprovedUNet3D(in_channels=1, out_channels=1, base_ch=32).to(device)
        print("🧠 Using ImprovedUNet3D (with Attention Gates)")
    else:
        model = UNet3D(in_channels=1, out_channels=1, base_ch=32).to(device)
        print("🧠 Using UNet3D (baseline)")

    # ---------- loss ----------
    if cfg["LOSS"].lower() == "hybrid":
        # Optionally set Focal alpha from data
        if cfg["FOCAL_ALPHA"] is None:
            alpha, fg_frac = _estimate_fg_alpha(tr_loader, max_batches=6)
            print(f"[info] Auto Focal alpha={alpha:.3f} (fg fraction ~{fg_frac:.3f})")
            # patch HybridLoss to use this alpha
            class _Hybrid(HybridLoss):
                def __init__(self): super().__init__()
                def forward(self, pred, target):
                    # swap Focal alpha on the fly
                    self.focal.alpha = alpha
                    return super().forward(pred, target)
            loss_fn = _Hybrid()
        else:
            loss_fn = HybridLoss()
        print("⚖️ Using HybridLoss (Dice + Focal)")
    else:
        loss_fn = DiceLoss()
        print("⚖️ Using DiceLoss only")

    # ---------- optim / sched ----------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       patience=cfg["LR_PATIENCE"], factor=cfg["LR_FACTOR"])

    # ---------- CSV log ----------
    csv_path = os.path.join(cfg["SAVE_DIR"], "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "lr"])

    best = 0.0
    hit_target_ep = None
    patience_counter = 0
    train_losses, val_dices = [], []

    print(f"Training for {cfg['EPOCHS']} epochs on {device} ...")
    for ep in range(1, cfg["EPOCHS"] + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, tr_loader, opt, loss_fn, device,
                                  use_amp=cfg["USE_AMP"], grad_clip=cfg["GRAD_CLIP"])
        val_dice = validate(model, va_loader, device, use_amp=False)
        sched.step(val_dice)

        train_losses.append(tr_loss); val_dices.append(val_dice)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.6f}", f"{val_dice:.6f}", f"{opt.param_groups[0]['lr']:.6f}"])

        # save example overlay from first val batch
        try:
            x0 = next(iter(va_loader))  # (safe: small batch)
            x = x0["image"].to(device); y = x0["mask"].to(device).unsqueeze(1).float()
            with torch.no_grad():
                p = model(x)
                y_ = _match_spatial_size(y, p, is_mask=True)
                p_ = _match_spatial_size(p, y_, is_mask=False)
                p_bin = (p_ > 0.5).float()
                _overlay_central_axial(x[0].cpu(), y_[0].cpu(), p_bin[0].cpu(),
                                       os.path.join(cfg["SAVE_DIR"], f"val_ep{ep:03d}.png"))
        except Exception as e:
            print(f"[viz warn] overlay failed: {e}")

        dt = time.time() - t0
        print(f"[{ep:03d}] loss={tr_loss:.4f} | val_dice={val_dice:.4f} | lr={opt.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        # checkpoint
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), os.path.join(cfg["SAVE_DIR"], "best.ckpt"))

        # early stop when target maintained for patience epochs
        if val_dice >= cfg["EARLY_STOP_TARGET"] - 1e-6:
            hit_target_ep = hit_target_ep or ep
            patience_counter = ep - hit_target_ep
            if patience_counter >= cfg["EARLY_STOP_PATIENCE"]:
                print(f"[early stop] Val Dice ≥ {cfg['EARLY_STOP_TARGET']} for {cfg['EARLY_STOP_PATIENCE']} epoch(s).")
                break
        else:
            hit_target_ep = None
            patience_counter = 0

    # curves
    try:
        plt.figure(); plt.plot(train_losses, label="train loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg["SAVE_DIR"], "loss_curve.png")); plt.close()
        plt.figure(); plt.plot(val_dices, label="val dice"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg["SAVE_DIR"], "dice_curve.png")); plt.close()
    except Exception as e:
        print(f"[plot warn] {e}")

    print(f"✅ Done. Best Val Dice = {best:.4f}")
    print(f"Logs & checkpoints → {cfg['SAVE_DIR']}")

if __name__ == "__main__":
    main()
