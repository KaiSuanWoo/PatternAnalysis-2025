import os, json, math, random
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from utils import check_cuda, set_seed, save_split
from dataset import find_patients, ProstatePatchDataset

def main():
    print("=== COMP3710 Project: 3D Improved U-Net ===")
    device = check_cuda()
    print(f"Running on: {device}")

if __name__ == "__main__":
    main()

DATA_ROOT = "Prostate3D_data"

def split_patients(root: str, seed: int = 1337,
                   train_ratio=0.7, val_ratio=0.15) -> Dict[str, List[str]]:
    items = find_patients(root)
    pids = [it.pid for it in items]
    random.Random(seed).shuffle(pids)
    n = len(pids); n_train = int(n*train_ratio); n_val = int(n*val_ratio)
    train_ids = pids[:n_train]
    val_ids   = pids[n_train:n_train+n_val]
    test_ids  = pids[n_train+n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}

def list_split(root: str) -> None:
    sp = split_patients(root)
    print("Patient split counts:", {k: len(v) for k,v in sp.items()})
    for k in ["train","val","test"]:
        print(f"{k}: {sp[k][:8]}{' ...' if len(sp[k])>8 else ''}")

def main():
    device = check_cuda()
    set_seed(1337)
    assert os.path.isdir(DATA_ROOT), f"DATA_ROOT not found: {DATA_ROOT}"

    # Print splits
    splits = split_patients(DATA_ROOT)
    print("Split sizes:", {k: len(v) for k,v in splits.items()})

    # Build datasets/loaders
    train_ds = ProstatePatchDataset(DATA_ROOT, splits["train"], patch_size=(128,128,128),
                                    foreground_prob=0.5, mode="train", augment=True)
    val_ds   = ProstatePatchDataset(DATA_ROOT, splits["val"], patch_size=(128,128,128),
                                    foreground_prob=0.0, mode="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Sanity batch
    batch = next(iter(train_loader))
    x, y = batch["image"], batch["mask"]
    print(f"Train batch shapes: image={tuple(x.shape)} mask={tuple(y.shape)} (device={device})")

if __name__ == "__main__":
    main()
