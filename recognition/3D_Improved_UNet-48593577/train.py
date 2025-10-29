import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    find_pairs,
    split_by_patient,
    Prostate3DDataset,
    NUM_CLASSES,
)

from modules import UNet3D


# -----------------------------
# Loss & metrics
# -----------------------------
class DiceCELoss(nn.Module):
    """
    CrossEntropy + Soft Dice (per class, averaged).
    Targets: (N, D, H, W) with class indices.
    Inputs:  (N, C, D, H, W) logits.
    """
    def __init__(self, num_classes, ce_weight=None, dice_weight=1.0, ce_weight_factor=1.0, eps=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        self.ce_weight_factor = ce_weight_factor
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target_idx):
        ce = self.ce(logits, target_idx) * self.ce_weight_factor

        # soft dice (ignore background? here we include all; change if needed)
        probs = torch.softmax(logits, dim=1)
        target_1h = torch.nn.functional.one_hot(target_idx.long(), num_classes=self.num_classes)
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()  # (N,C,D,H,W)

        dims = (0, 2, 3, 4)
        intersect = torch.sum(probs * target_1h, dims)
        denom = torch.sum(probs + target_1h, dims)
        dice_per_class = (2 * intersect + self.eps) / (denom + self.eps)
        dice = dice_per_class.mean()

        loss = ce + self.dice_weight * (1 - dice)
        return loss, dice, dice_per_class.detach()


@torch.no_grad()
def eval_epoch(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_denom = torch.zeros(num_classes, device=device)

    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        # labels from dataset: index map if one_hot=False
        labels = labels.to(device, non_blocking=True).long()

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)

        # CE
        loss = ce(logits, labels)

        # Soft dice accumulation
        tgt_1h = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        inter = torch.sum(probs * tgt_1h, dim=(0, 2, 3, 4))
        denom = torch.sum(probs + tgt_1h, dim=(0, 2, 3, 4))
        per_class_inter += inter
        per_class_denom += denom

        total_loss += loss.item()
        count += 1

    dice_per_class = (2 * per_class_inter + 1e-6) / (per_class_denom + 1e-6)
    mean_dice = dice_per_class.mean().item() if count > 0 else 0.0
    return total_loss / max(count, 1), mean_dice, dice_per_class.detach().cpu().numpy()


# -----------------------------
# Training
# -----------------------------
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    pairs = find_pairs()
    train_pairs, val_pairs = split_by_patient(pairs, train_ratio=args.train_ratio, seed=args.seed)

    train_ds = Prostate3DDataset(train_pairs, target_dhw=tuple(args.target), one_hot=False)
    val_ds   = Prostate3DDataset(val_pairs,   target_dhw=tuple(args.target), one_hot=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch,
        depth=args.depth,
        use_se=not args.no_se,
        use_att=not args.no_att,
        dropout=args.dropout
    ).to(device)

    # Optimizer / Scheduler / Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    # Loss
    loss_fn = DiceCELoss(num_classes=NUM_CLASSES, dice_weight=args.dice_w, ce_weight_factor=args.ce_w)

    # Logging
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_D{args.depth}_B{args.base_ch}_{args.target}"
    ckpt_best = os.path.join(args.save_dir, f"{run_name}_best.pt")
    log_path = os.path.join(args.save_dir, f"{run_name}_log.jsonl")
    best_dice = -1.0

    print(f"Training on {device} | NUM_CLASSES={NUM_CLASSES} | target={args.target}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        running_loss = 0.0
        running_dice = 0.0
        batches = 0

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()  # (N,D,H,W)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(imgs)  # (N,C,D,H,W)
                loss, dice, _ = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_dice += dice.item()
            batches += 1
            pbar.set_postfix(loss=f"{running_loss/batches:.4f}", dice=f"{running_dice/batches:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        sched.step()

        # Eval
        val_loss, val_dice, dice_classes = eval_epoch(model, val_loader, device, NUM_CLASSES)

        log_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "train_dice": running_dice / max(batches, 1),
            "val_loss": val_loss,
            "val_dice": val_dice,
            "dice_per_class": dice_classes.tolist(),
            "lr": sched.get_last_lr()[0],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

        print(f"Epoch {epoch}: val_dice={val_dice:.4f} | per-class={np.round(dice_classes,4)}")

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "num_classes": NUM_CLASSES
            }, ckpt_best)
            print(f"✅ Saved best checkpoint to {ckpt_best} (dice={best_dice:.4f})")

    print("Training finished.")


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Train Improved 3D U-Net on Prostate3D")
    # Model
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--no_se", action="store_true", help="disable SE channel attention in blocks")
    p.add_argument("--no_att", action="store_true", help="disable attention gates on skips")
    p.add_argument("--dropout", type=float, default=0.0)

    # Data
    p.add_argument("--target", type=int, nargs=3, default=[96, 192, 192], help="D H W")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)

    # Optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--cpu", action="store_true")

    # Loss weights
    p.add_argument("--dice_w", type=float, default=1.0)
    p.add_argument("--ce_w", type=float, default=1.0)

    # IO
    p.add_argument("--save_dir", type=str, default="runs")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.backends.cudnn.benchmark = True
    train(args)
