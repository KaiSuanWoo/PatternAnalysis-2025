from dataset import find_pairs, split_by_patient, Prostate3DDataset, NUM_CLASSES

pairs = find_pairs()  # or pass custom paths
train_pairs, val_pairs = split_by_patient(pairs, train_ratio=0.8, seed=1337)

target_size = (96, 192, 192)  # (D,H,W) — pick what fits your GPU

train_ds = Prostate3DDataset(train_pairs, target_dhw=target_size, one_hot=True)
val_ds   = Prostate3DDataset(val_pairs,   target_dhw=target_size, one_hot=True)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

print("NUM_CLASSES:", NUM_CLASSES)  # feed to your 3D UNet head
