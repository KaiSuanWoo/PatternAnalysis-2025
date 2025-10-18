import torch
import os, math, random, json, re
from typing import Tuple, List, Dict, Optional
import numpy as np

def check_cuda():
    """Print and return active device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device("cpu")

def set_seed(seed: int = 1337) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def percentile_clip_zscore(vol: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Clip intensities to [p_lo, p_hi] and z-score using nonzero voxels."""
    assert vol.ndim == 3
    nz = vol[vol != 0]
    if nz.size == 0:
        return vol.astype(np.float32)
    lo, hi = np.percentile(nz, [p_lo, p_hi])
    vol = np.clip(vol, lo, hi)
    mu, sd = nz.mean(), nz.std() + 1e-8
    return ((vol - mu) / sd).astype(np.float32)

def random_crop3d(img: np.ndarray, msk: np.ndarray, size: Tuple[int,int,int]) -> Tuple[np.ndarray,np.ndarray]:
    D,H,W = img.shape
    d,h,w = size
    assert D>=d and H>=h and W>=w, f"Patch {size} larger than vol {img.shape}"
    z = np.random.randint(0, D-d+1)
    y = np.random.randint(0, H-h+1)
    x = np.random.randint(0, W-w+1)
    return img[z:z+d, y:y+h, x:x+w], msk[z:z+d, y:y+h, x:x+w]

def has_foreground(msk_patch: np.ndarray) -> bool:
    return np.count_nonzero(msk_patch) > 0

def to_tensor(img: np.ndarray, msk: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # add channel dim → (C=1, D, H, W)
    img_t = torch.from_numpy(img)[None, ...]  # float32
    msk_t = torch.from_numpy(msk.astype(np.int64))  # (D,H,W) long
    return img_t, msk_t

def save_split(paths: Dict[str, List[str]], out_path: str) -> None:
    with open(out_path, "w") as f: json.dump(paths, f, indent=2)
