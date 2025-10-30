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

    Returns:
        loss (scalar),
        dice_mean (scalar),
        dice_per_class (C,) tensor (no grad).
    """
    def __init__(
        self,
        num_classes,
        ce_weight=None,
        dice_weight=1.0,
        ce_weight_factor=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight
        self.ce_weight_factor = ce_weight_factor
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target_idx):
        # logits: (N,C,D,H,W), target_idx: (N,D,H,W)
        ce = self.ce(logits, target_idx) * self.ce_weight_factor

        probs = torch.softmax(logits, dim=1)  # (N,C,D,H,W)

        # one-hot of target for dice calc
        target_1h = torch.nn.functional.one_hot(
            target_idx.long(),
            num_classes=self.num_classes
        )  # (N,D,H,W,C)
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()  # (N,C,D,H,W)

        dims = (0, 2, 3, 4)
        intersect = torch.sum(probs * target_1h, dims)          # (C,)
        denom     = torch.sum(probs + target_1h, dims)           # (C,)
        dice_per_class = (2 * intersect + self.eps) / (denom + self.eps)  # (C,)

        dice_mean = dice_per_class.mean()  # scalar
        loss = ce + self.dice_weight * (1 - dice_mean)

        return loss, dice_mean, dice_per_class.detach()


@torch.no_grad()
def eval_epoch(model, loader, device, num_classes):
    """
    Run validation:
    - returns (val_loss_scalar, val_mean_dice_scalar, val_dice_per_class_np)
    """
    model.eval()
    total_loss = 0.0
    total_dice_mean = 0.0
    count = 0

    ce = nn.CrossEntropyLoss()

    # accumulate dice across val set
    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_denom = torch.zeros(num_classes, device=device)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()  # (N,D,H,W)

        logits = model(imgs)                   # (N,C,D,H,W)
        probs = torch.softmax(logits, dim=1)   # (N,C,D,H,W)

        # CE component for reference
        loss = ce(logits, labels)

        # 1-hot GT for dice
        tgt_1h = torch.nn.functional.one_hot(
            labels,
            num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()      # (N,C,D,H,W)

        inter = torch.sum(probs * tgt_1h, dim=(0, 2, 3, 4))   # (C,)
        denom = torch.sum(probs + tgt_1h, dim=(0, 2, 3, 4))   # (C,)
        per_class_inter += inter
        per_class_denom += denom

        total_loss += loss.item()
        count += 1

    dice_per_class = (2 * per_class_inter + 1e-6) / (per_class_denom + 1e-6)  # (C,)
    mean_dice = dice_per_class.mean().item() if count > 0 else 0.0
    mean_loss = total_loss / max(count, 1)

    return mean_loss, mean_dice, dice_per_class.detach().cpu().numpy()


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------
    # Dataset split
    # -----------------
    pairs = find_pairs()
    train_pairs, val_pairs = split_by_patient(
        pairs,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    train_ds = Prostate3DDataset(
        train_pairs,
        target_dhw=tuple(args.target),
        one_hot=False
    )
    val_ds = Prostate3DDataset(
        val_pairs,
        target_dhw=tuple(args.target),
        one_hot=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # -----------------
    # Model & device
    # -----------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = UNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch,
        depth=args.depth,
        use_se=not args.no_se,
        use_att=not args.no_att,
        dropout=args.dropout
    ).to(device)

    # -----------------
    # Optimizer / Scheduler / AMP
    # -----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.amp)
    )

    # -----------------
    # Loss fn (train side)
    # -----------------
    loss_fn = DiceCELoss(
        num_classes=NUM_CLASSES,
        dice_weight=args.dice_w,
        ce_weight_factor=args.ce_w
    )

    # -----------------
    # Logging setup
    # -----------------
    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"_D{args.depth}_B{args.base_ch}_{args.target}"
    )
    ckpt_best = os.path.join(args.save_dir, f"{run_name}_best.pt")
    log_path = os.path.join(args.save_dir, f"{run_name}_log.jsonl")

    best_dice = -1.0

    print(
        f"Training on {device} | NUM_CLASSES={NUM_CLASSES} | "
        f"target={args.target}"
    )

    # -----------------
    # Epoch loop
    # -----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=100
        )

        running_loss = 0.0
        running_dice = 0.0
        batches = 0

        # We'll average per-class Dice across training batches
        running_dice_per_class_sum = torch.zeros(
            NUM_CLASSES,
            device=device
        )
        running_dice_per_class_count = 0

        # -------- train batches --------
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()  # (N,D,H,W)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda" and args.amp)
            ):
                logits = model(imgs)  # (N,C,D,H,W)
                loss, dice_mean, dice_pc = loss_fn(logits, labels)
                # dice_pc: per-class Dice for THIS batch (C,)

            scaler.scale(loss).backward()

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.clip_grad
                )

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_dice += dice_mean.item()
            batches += 1

            running_dice_per_class_sum += dice_pc.to(device)
            running_dice_per_class_count += 1

            pbar.set_postfix(
                loss=f"{running_loss/batches:.4f}",
                dice=f"{running_dice/batches:.4f}",
                lr=f"{sched.get_last_lr()[0]:.2e}"
            )

        # step LR after each epoch
        sched.step()

        # -------- aggregate train stats for epoch --------
        train_loss_epoch = running_loss / max(batches, 1)
        train_dice_epoch = running_dice / max(batches, 1)

        if running_dice_per_class_count > 0:
            train_dice_classes_epoch = (
                running_dice_per_class_sum /
                running_dice_per_class_count
            ).detach().cpu().numpy()
        else:
            train_dice_classes_epoch = np.zeros(NUM_CLASSES, dtype=np.float32)

        # -------- validation --------
        val_loss, val_dice, val_dice_classes = eval_epoch(
            model,
            val_loader,
            device,
            NUM_CLASSES
        )
        # val_dice_classes: np.array(C,)

        # -------- log row (goes to jsonl used by predict.py plots) --------
        log_row = {
            "epoch": epoch,

            # scalar curves
            "train_loss": float(train_loss_epoch),
            "train_dice": float(train_dice_epoch),
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),

            # per-class curves for training/validation dice
            "train_dice_classes": train_dice_classes_epoch.tolist(),
            "val_dice_classes":   val_dice_classes.tolist(),

            # learning rate snapshot
            "lr": float(sched.get_last_lr()[0]),
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

        print(
            f"Epoch {epoch}: "
            f"train_dice={train_dice_epoch:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_per_class={np.round(val_dice_classes,4)}"
        )

        # -------- checkpointing best model --------
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "num_classes": NUM_CLASSES
            }, ckpt_best)
            print(
                f"✅ Saved best checkpoint to {ckpt_best} "
                f"(dice={best_dice:.4f})"
            )

    print("Training finished.")


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Train Improved 3D U-Net on Prostate3D"
    )

    # Model
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--no_se", action="store_true",
                   help="disable SE channel attention in blocks")
    p.add_argument("--no_att", action="store_true",
                   help="disable attention gates on skips")
    p.add_argument("--dropout", type=float, default=0.0)

    # Data / split
    p.add_argument(
        "--target",
        type=int,
        nargs=3,
        default=[96, 192, 192],
        help="D H W input volume size (will be cropped/reshaped)"
    )
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)

    # Optim
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true",
                   help="use mixed precision if CUDA is available")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU mode even if CUDA available")

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
