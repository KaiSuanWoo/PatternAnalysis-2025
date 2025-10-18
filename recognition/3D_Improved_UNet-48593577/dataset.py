from __future__ import annotations
import os, glob
from typing import List, Tuple, Dict, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# ---------- helpers (local to this file) ----------

def _as_canonical(img_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """Re-orient to RAS+ (closest canonical)."""
    return nib.as_closest_canonical(img_nii)

def _percentile_clip_zscore(vol: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    nz = vol[vol != 0]
    if nz.size == 0:
        return vol.astype(np.float32)
    lo, hi = np.percentile(nz, [p_lo, p_hi])
    vol = np.clip(vol, lo, hi)
    mu, sd = nz.mean(), nz.std() + 1e-8
    return ((vol - mu) / sd).astype(np.float32)

def _ensure_3d(x: np.ndarray) -> np.ndarray:
    """
    Some medical files arrive as 4D with a singleton or sequence dim (D,H,W,1 or D,H,W,T).
    We keep the first volume (Week/LFOV series) and ensure shape (D,H,W).
    """
    if x.ndim == 4:
        x = x[..., 0]
    assert x.ndim == 3, f"Expected 3D volume, got shape {x.shape}"
    return x

def _to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """(D,H,W) int -> (C,D,H,W) one-hot."""
    out = np.zeros((num_classes, *mask.shape), dtype=np.uint8)
    for c in range(num_classes):
        out[c] = (mask == c).astype(np.uint8)
    return out

# ---------- matched pair discovery (unchanged logic, just utility here) ----------

class Prostate3DRawItem:
    def __init__(self, pid: str, img_path: str, msk_path: str):
        self.pid = pid
        self.img_path = img_path
        self.msk_path = msk_path

def find_patients(root: str) -> List[Prostate3DRawItem]:
    imgs_dir = os.path.join(root, "semantic_MRs_anon")
    labs_dir = os.path.join(root, "semantic_labels_anon")
    assert os.path.isdir(imgs_dir) and os.path.isdir(labs_dir), "Missing expected subfolders!"

    imgs = sorted(glob.glob(os.path.join(imgs_dir, "*.nii*")))
    labs = sorted(glob.glob(os.path.join(labs_dir, "*.nii*")))

    def key(p: str) -> str:
        base = os.path.basename(p)
        k = base.split("_LFOV")[0]
        k = k.replace("_SEMANTIC", "")
        return k

    img_map = {key(p): p for p in imgs}
    lab_map = {key(p): p for p in labs}
    keys = sorted(set(img_map) & set(lab_map))

    items = [Prostate3DRawItem(k, img_map[k], lab_map[k]) for k in keys]
    if not items:
        raise FileNotFoundError("No matched MRI/label pairs found.")
    print(f"Found {len(items)} matched MRI/label pairs")
    return items

# ---------- NEW: load_data_3D-style bulk loader ----------

def load_data_3D(
    imageNames: List[str],
    normImage: bool = False,
    categorical: bool = False,
    dtype: np.dtype = np.float32,
    getAffines: bool = False,
    orient: bool = False,
    early_stop: bool = False,
    label_mode: bool = False,
    num_classes: Optional[int] = None,
):
    """
    Load medical image data from file paths into a single numpy array
    (pre-allocates array for conv3d efficiency), closely following the example you provided.

    - normImage: normalise each image to z-score in non-zero voxels after percentile clip.
    - orient: re-orient to RAS+ using nibabel.as_closest_canonical (no resample of spacing).
    - dtype: np.float32 for images; if labels, we override to uint8.
    - early_stop: if True, stop after ~20 cases (useful for quick scripts).
    - categorical: if True (for labels), output one-hot (C,D,H,W) using num_classes.
    - label_mode: set True when loading labels; affects dtype handling.
    - getAffines: if True, also return list of affines.
    """
    affines = []
    interp = "nearest" if (label_mode or dtype == np.uint8) else "linear"

    # determine common shape by reading first case
    first = nib.load(imageNames[0])
    if orient:
        first = _as_canonical(first)
    first_arr = first.get_fdata(caching="unchanged")
    first_arr = _ensure_3d(first_arr)
    if categorical:
        assert num_classes is not None, "num_classes required for categorical labels"
        images = np.zeros((len(imageNames), num_classes, *first_arr.shape), dtype=np.uint8)
    else:
        images = np.zeros((len(imageNames), *first_arr.shape), dtype=(np.uint8 if label_mode else dtype))

    for i, path in enumerate(imageNames):
        nii = nib.load(path)
        if orient:
            nii = _as_canonical(nii)
        arr = nii.get_fdata(caching="unchanged")
        arr = _ensure_3d(arr)

        # clip depth if shape mismatch happens across cases (rare)
        arr = arr[: first_arr.shape[0], : first_arr.shape[1], : first_arr.shape[2]]

        if normImage and not label_mode:
            arr = _percentile_clip_zscore(arr)

        if label_mode:
            arr = arr.astype(np.uint8)
            # binarize if labels are >1 (semantic labels → foreground OR)
            if arr.max() > 1 and not categorical:
                arr = (arr > 0).astype(np.uint8)

        if categorical:
            oh = _to_onehot(arr.astype(np.int16), num_classes=num_classes)
            images[i] = oh
        else:
            images[i] = arr.astype(images.dtype)

        affines.append(nii.affine)

        if early_stop and i > 20:
            break

    if getAffines:
        return images, affines
    return images

# ---------- Dataset class leveraging the upgraded loader ----------

class ProstatePatchDataset(Dataset):
    """
    Patch-based dataset for training/validation, built on top of load_data_3D-like behaviour.
    When mode='test', you generally want to load full volumes and use sliding-window inference elsewhere.
    """
    def __init__(
        self,
        root: str,
        ids: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        foreground_prob: float = 0.5,
        mode: str = "train",
        augment: bool = True,
        normImage: bool = True,
        orient: bool = True,
        categorical_labels: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.root = root
        self.ids = ids
        self.patch = patch_size
        self.fg_prob = foreground_prob
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.normImage = normImage
        self.orient = orient
        self.categorical = categorical_labels
        self.num_classes = num_classes

        # map id -> paths
        all_items = {it.pid: it for it in find_patients(root)}
        self.items = [all_items[i] for i in ids]

    def __len__(self) -> int:
        return max(1, len(self.ids) * 16)

    # --- light aug (keep robust) ---
    def _augment(self, img: np.ndarray, msk: np.ndarray):
        if np.random.rand() < 0.5:
            img = img[::-1, :, :]; msk = msk[::-1, :, :]
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]; msk = msk[:, ::-1, :]
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1]; msk = msk[:, :, ::-1]
        if np.random.rand() < 0.3:
            img = img * (0.95 + 0.1*np.random.rand()) + np.random.uniform(-0.05, 0.05)
        return img, msk

    def _load_pair(self, item: Prostate3DRawItem) -> Tuple[np.ndarray, np.ndarray]:
        # single-file list to reuse load_data_3D
        imgs = load_data_3D(
            [item.img_path],
            normImage=self.normImage,
            categorical=False,
            dtype=np.float32,
            getAffines=False,
            orient=self.orient,
            early_stop=False,
            label_mode=False,
        )[0]  # -> (D,H,W)

        labs = load_data_3D(
            [item.msk_path],
            normImage=False,
            categorical=False,   # set True if you want one-hot
            dtype=np.uint8,
            getAffines=False,
            orient=self.orient,
            early_stop=False,
            label_mode=True,
            num_classes=self.num_classes,
        )[0]

        # ensure binary for baseline
        if labs.max() > 1 and not self.categorical:
            labs = (labs > 0).astype(np.uint8)

        return imgs, labs

    def _rand_crop3d(self, img: np.ndarray, msk: np.ndarray, size: Tuple[int,int,int]):
        D, H, W = img.shape
        d, h, w = size
        dz = np.random.randint(0, max(1, D - d + 1))
        dy = np.random.randint(0, max(1, H - h + 1))
        dx = np.random.randint(0, max(1, W - w + 1))
        return img[dz:dz+d, dy:dy+h, dx:dx+w], msk[dz:dz+d, dy:dy+h, dx:dx+w]

    def __getitem__(self, idx: int):
        item = self.items[idx % len(self.items)]
        img, msk = self._load_pair(item)

        # foreground-biased patch sampling (train only)
        want_fg = (np.random.rand() < self.fg_prob) and (self.mode == "train")
        for _ in range(16):
            p_img, p_msk = self._rand_crop3d(img, msk, self.patch)
            if (not want_fg) or (p_msk.sum() > 0):
                img, msk = p_img, p_msk
                break

        if self.augment:
            img, msk = self._augment(img, msk)

        # tensors
        img_t = torch.from_numpy(img.astype(np.float32))[None, ...]           # (1,D,H,W)
        msk_t = torch.from_numpy(msk.astype(np.int64))                        # (D,H,W)
        return {"image": img_t, "mask": msk_t, "pid": item.pid}
