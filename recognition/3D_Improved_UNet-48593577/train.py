import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import UNet3D
from dataset import ProstatePatchDataset, find_patients
from utils import check_cuda, set_seed, DiceLoss, dice_coefficient

DATA_ROOT = "Prostate3D_data"
SAVE_DIR = "runs"
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1
EPOCHS = 20
TRAIN_RATIO = 0.8
SEED = 1337


def split_ids(root: str, train_ratio: float = TRAIN_RATIO, seed: int = SEED):
    patients = find_patients(root)
    ids = [p.pid for p in patients]
    if not ids:
        raise ValueError("No patient IDs found for training.")

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_train = max(1, int(len(ids) * train_ratio))
    if len(ids) > 1:
        n_train = min(len(ids) - 1, n_train)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    if not val_ids:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1] or train_ids

    return train_ids, val_ids


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    iterator = tqdm(loader, desc="Train", leave=False)

    for batch in iterator:
        imgs = batch["image"].to(device)  # (N,1,D,H,W)
        masks = batch["mask"].float().unsqueeze(1).to(device)  # (N,1,D,H,W)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            loss = loss_fn(preds, masks)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(1, len(loader))


def validate(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"].float().unsqueeze(1).to(device)
            preds = model(imgs)
            dice = dice_coefficient(preds, masks).item()
            scores.append(dice)
    return sum(scores) / max(1, len(scores))


def main():
    device = check_cuda()
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_ids, val_ids = split_ids(DATA_ROOT, TRAIN_RATIO, SEED)
    print(f"Split sizes → train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = ProstatePatchDataset(DATA_ROOT, train_ids, patch_size=PATCH_SIZE,
                                    foreground_prob=0.5, mode="train", augment=True)
    val_ds = ProstatePatchDataset(DATA_ROOT, val_ids, patch_size=PATCH_SIZE,
                                  foreground_prob=0.0, mode="val", augment=False)

    num_workers = 2 if len(train_ds) > 0 else 0
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent_workers)

    model = UNet3D().to(device)
    loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=use_amp) if use_amp else None

    best_dice = 0.0
    epoch_iter = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, use_amp)
        val_dice = validate(model, val_loader, device)
        dt = time.time() - t0

        epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}", val_dice=f"{val_dice:.4f}", time=f"{dt:.1f}s")

        if val_dice >= best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.ckpt"))

    print(f"✅ Training complete. Best Val Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
