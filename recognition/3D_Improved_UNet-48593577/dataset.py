from __future__ import annotations
import os, glob
from typing import List, Tuple, Dict, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import percentile_clip_zscore, random_crop3d, has_foreground, to_tensor

class Prostate3DRawItem:
    def __init__(self, pid: str, img_path: str, msk_path: str):
        self.pid = pid
        self.img_path = img_path
        self.msk_path = msk_path

def find_patients(root: str) -> List[Prostate3DRawItem]:
    """Match MRI and label files from the two subfolders by case name."""
    imgs_dir = os.path.join(root, "semantic_MRs_anon")
    labs_dir = os.path.join(root, "semantic_labels_anon")
    assert os.path.isdir(imgs_dir) and os.path.isdir(labs_dir), "Missing expected subfolders!"

    imgs = sorted(glob.glob(os.path.join(imgs_dir, "*.nii.gz")))
    labs = sorted(glob.glob(os.path.join(labs_dir, "*.nii.gz")))

    items = []
    # Extract shared prefix (e.g. Case_004_Week0)
    def extract_key(path: str) -> str:
        base = os.path.basename(path)
        key = base.split("_LFOV")[0]
        key = key.replace("_SEMANTIC", "")  # remove label suffix if present
        return key

    img_dict = {extract_key(p): p for p in imgs}
    lab_dict = {extract_key(p): p for p in labs}

    shared_keys = sorted(set(img_dict.keys()) & set(lab_dict.keys()))
    print(f"Found {len(shared_keys)} matched MRI/label pairs")

    for key in shared_keys:
        items.append(Prostate3DRawItem(pid=key,
                                       img_path=img_dict[key],
                                       msk_path=lab_dict[key]))
    if not items:
        raise FileNotFoundError("No matched image/label pairs found.")
    return items


def load_nii(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)

def load_label_nii(path: str) -> np.ndarray:
    # ensure integer mask (0/1 or 0..K)
    data = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    return data.astype(np.int16)

class ProstatePatchDataset(Dataset):
    """
    Pytorch dataset backed by the raw Prostate3D folders (semantic_MRs_anon / semantic_labels_anon).
    Files are matched by shared prefix, intensities z-scored using 1-99th percentiles,
    and 128^3 patches are sampled with optional foreground bias for training.
    """
    def __init__(self,
                 root: str,
                 ids: Optional[List[str]] = None,
                 patch_size: Tuple[int,int,int] = (128,128,128),
                 foreground_prob: float = 0.5,
                 mode: str = "train",
                 augment: bool = True):
        super().__init__()
        self.root = root
        self.patch = patch_size
        self.fg_prob = foreground_prob
        self.mode = mode
        self.augment = augment and (mode == "train")

        all_items = {item.pid: item for item in find_patients(root)}
        if ids is None:
            ids = sorted(all_items.keys())
        missing = sorted(set(ids) - set(all_items.keys()))
        if missing:
            raise ValueError(f"Patient IDs not found in dataset root: {missing}")

        self.ids = list(ids)
        self.items: Dict[str, Prostate3DRawItem] = {pid: all_items[pid] for pid in self.ids}
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        # arbitrary epoch length; you can scale with number of patients
        return max(1, len(self.ids) * 16)

    def _load_pair(self, item: Prostate3DRawItem) -> Tuple[np.ndarray, np.ndarray]:
        if item.pid in self._cache:
            return self._cache[item.pid]

        img = load_nii(item.img_path)    # (D,H,W) float32
        msk = load_label_nii(item.msk_path)  # (D,H,W) int
        img = percentile_clip_zscore(img, 1.0, 99.0)
        # Cast mask to {0,1} if necessary
        if msk.max() > 1:
            msk = (msk > 0).astype(np.int16)

        self._cache[item.pid] = (img, msk)
        return img, msk

    def _augment_inplace(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        # light 3D flips and small rotations (rotations omitted for simplicity/robustness)
        if np.random.rand() < 0.5:
            img = img[::-1, :, :]; msk = msk[::-1, :, :]
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]; msk = msk[:, ::-1, :]
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1]; msk = msk[:, :, ::-1]
        # mild intensity jitter
        if np.random.rand() < 0.3:
            img = img * (0.95 + 0.1*np.random.rand()) + np.random.uniform(-0.05, 0.05)
        return img, msk

    def __getitem__(self, idx: int):
        # choose a patient
        pid = self.ids[idx % len(self.ids)]
        item = self.items[pid]
        img, msk = self._load_pair(item)

        # sample patch (optionally enforce foreground with probability fg_prob)
        want_fg = (np.random.rand() < self.fg_prob) and (self.mode == "train")
        for _ in range(16):  # try up to 16 times to find a fg patch
            p_img, p_msk = random_crop3d(img, msk, self.patch)
            if (not want_fg) or has_foreground(p_msk):
                img, msk = p_img, p_msk
                break

        if self.augment:
            img, msk = self._augment_inplace(img, msk)

        img = np.ascontiguousarray(img)
        msk = np.ascontiguousarray(msk)

        img_t, msk_t = to_tensor(img, msk)  # (1,D,H,W), (D,H,W)
        return {"image": img_t, "mask": msk_t, "pid": pid}
