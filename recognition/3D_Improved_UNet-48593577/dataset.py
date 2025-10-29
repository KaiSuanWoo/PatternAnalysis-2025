import os
import socket
import re
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------
# Dataset path setup — defaults to Rangpur but works locally
# -----------------------------------------------------

# Rangpur cluster dataset
RANGPUR_ROOT  = "/home/groups/comp3710/HipMRI_Study_open"
RANGPUR_IMGS  = os.path.join(RANGPUR_ROOT, "semantic_MRs")
RANGPUR_LABELS = os.path.join(RANGPUR_ROOT, "semantic_labels_only")

# Local fallback (for debugging / testing)
LOCAL_ROOT = os.path.join(os.path.dirname(__file__), "Prostate3D_data")
LOCAL_IMGS = os.path.join(LOCAL_ROOT, "semantic_MRs_anon")
LOCAL_LABELS = os.path.join(LOCAL_ROOT, "semantic_labels_anon")

# Automatically pick Rangpur if it exists, else local
if os.path.exists(RANGPUR_IMGS) and os.path.exists(RANGPUR_LABELS):
    DATA_ROOT  = RANGPUR_ROOT
    IMAGES_DIR = RANGPUR_IMGS
    LABELS_DIR = RANGPUR_LABELS
    print(f"[dataset] Using Rangpur dataset at: {DATA_ROOT}")
else:
    DATA_ROOT  = LOCAL_ROOT
    IMAGES_DIR = LOCAL_IMGS
    LABELS_DIR = LOCAL_LABELS
    print(f"[dataset] Using local dataset at: {DATA_ROOT}")

# -----------------------------
# Filename parsing
# -----------------------------
# Examples:
#   MRI:   Case_004_Week0_LFOV.nii.gz
#   Label: Case_004_Week0_SEMANTIC_LFOV.nii.gz
MRI_RE   = re.compile(r"^(Case_\d+)_Week(\d+)_LFOV\.nii\.gz$")
LABEL_RE = re.compile(r"^(Case_\d+)_Week(\d+)_SEMANTIC_LFOV\.nii\.gz$")


def parse_mri_name(fname: str):
    m = MRI_RE.match(fname)
    if not m:
        return None
    case, week = m.group(1), int(m.group(2))
    return case, week


def expected_label_name_from_mri(fname: str) -> str:
    # Case_004_Week0_LFOV.nii.gz  ->  Case_004_Week0_SEMANTIC_LFOV.nii.gz
    return fname.replace("_LFOV.nii.gz", "_SEMANTIC_LFOV.nii.gz")


# -----------------------------
# I/O helpers
# -----------------------------
def read_nifti(path: str, orient: bool = True):
    img = nib.load(path)
    if orient:
        img = nib.as_closest_canonical(img)
    arr = img.get_fdata(caching="unchanged")
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr, img.affine


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + eps)


def pad_or_crop_center(x: np.ndarray, target: Tuple[int, int, int], pad_value: float = 0):
    """Center pad/crop a 3D array to target (D,H,W)."""
    assert x.ndim == 3
    D, H, W = x.shape
    TD, TH, TW = target

    # pad
    pd = max(TD - D, 0); ph = max(TH - H, 0); pw = max(TW - W, 0)
    if pd or ph or pw:
        x = np.pad(
            x,
            ((pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
            mode="constant",
            constant_values=pad_value,
        )
        D, H, W = x.shape

    # crop
    sd = max((D - TD) // 2, 0)
    sh = max((H - TH) // 2, 0)
    sw = max((W - TW) // 2, 0)
    x = x[sd:sd + TD, sh:sh + TH, sw:sw + TW]
    return x


def build_label_map(unique_vals: np.ndarray, include_background: bool = True) -> Dict[int, int]:
    unique_vals = np.sort(unique_vals.astype(int))
    lm: Dict[int, int] = {}
    k = 0
    if include_background and 0 in unique_vals:
        lm[0] = 0
        k = 1
    for v in unique_vals:
        v = int(v)
        if include_background and v == 0:
            continue
        lm[v] = k
        k += 1
    return lm


def to_one_hot(lbl: np.ndarray, label_map: Dict[int, int], dtype=np.uint8) -> np.ndarray:
    assert lbl.ndim == 3
    C = len(set(label_map.values()))
    oh = np.zeros((C,) + lbl.shape, dtype=dtype)
    for raw, idx in label_map.items():
        oh[idx][lbl == raw] = 1
    return oh


# -----------------------------
# Pair discovery (robust to names)
# -----------------------------
# Accept keys like: B006_Week0_*.nii.gz  OR  Case_004_Week6_*.nii.gz
KEY_RE = re.compile(r"^((?:Case_\d+)|(?:[A-Za-z]\d+))_Week(\d+).+\.nii\.gz$", re.IGNORECASE)


def _index_by_key(dir_path: str) -> dict:
    idx = {}
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".nii.gz")])
    for f in files:
        m = KEY_RE.match(f)
        if not m:
            continue
        case, week = m.group(1), int(m.group(2))
        idx[(case, week)] = os.path.join(dir_path, f)
    return idx

def find_pairs(images_dir: str = IMAGES_DIR,
               labels_dir: str = LABELS_DIR) -> List[Tuple[str, str, str, int]]:
    """
    Returns a list of (img_path, lbl_path, patient_id, week),
    matched by (Case_xxx, WeekN). Robust to extra tokens like LFOV/SEMANTIC/etc.
    """
    img_idx = _index_by_key(images_dir)
    lbl_idx = _index_by_key(labels_dir)

    # intersect keys
    keys = sorted(set(img_idx.keys()) & set(lbl_idx.keys()))
    pairs = [(img_idx[k], lbl_idx[k], k[0], k[1]) for k in keys]

    if not pairs:
        # Helpful diagnostics
        imgs = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])[:5]
        lbls = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])[:5]
        raise RuntimeError(
            "No (image,label) pairs found by (Case_xxx, WeekN) key.\n"
            f"Checked folders:\n  images_dir={images_dir}\n  labels_dir={labels_dir}\n"
            f"Example image files: {imgs}\nExample label files: {lbls}\n"
            "Ensure filenames look like 'Case_004_Week6_*.nii.gz'."
        )
    return pairs


def split_by_patient(pairs: List[Tuple[str, str, str, int]],
                     train_ratio: float = 0.8,
                     seed: int = 1337):
    """
    Split by patient (Case_xxx), keeping all weeks together.
    Returns (train_pairs, val_pairs).
    """
    rng = random.Random(seed)
    patients = sorted({p[2] for p in pairs})
    rng.shuffle(patients)

    n_train = max(1, int(len(patients) * train_ratio))
    if len(patients) > 1:
        n_train = min(len(patients) - 1, n_train)

    train_patients = set(patients[:n_train])
    train_pairs = [p for p in pairs if p[2] in train_patients]
    val_pairs   = [p for p in pairs if p[2] not in train_patients]
    return train_pairs, val_pairs


def scan_label_values(pairs: List[Tuple[str, str, str, int]], max_files: Optional[int] = None) -> List[int]:
    """
    Inspect labels to get the set of unique class ids.
    """
    uniq = set()
    for i, (_, lbl_path, _, _) in enumerate(pairs):
        if max_files is not None and i >= max_files:
            break
        lab, _ = read_nifti(lbl_path, orient=True)
        if lab.ndim == 4 and lab.shape[-1] == 1:
            lab = lab[..., 0]
        uniq.update(np.unique(lab.astype(int)).tolist())
    return sorted(list(uniq))


# -----------------------------
# PyTorch Dataset
# -----------------------------
class Prostate3DDataset(Dataset):
    """
    Yields (image, label) tensors.
      image: (1, D, H, W)  float32  (z-score normalized)
      label: (C, D, H, W)  uint8    if one_hot=True
             (D, H, W)     uint8    if one_hot=False (class indices)
    """
    def __init__(self,
                 pairs: List[Tuple[str, str, str, int]],
                 target_dhw: Optional[Tuple[int, int, int]] = None,
                 one_hot: bool = True,
                 known_label_values: Optional[List[int]] = None):
        super().__init__()
        self.pairs = pairs
        self.target = target_dhw
        self.one_hot = one_hot

        # Build label map
        if known_label_values is None:
            values = scan_label_values(pairs, max_files=None)
        else:
            values = known_label_values
        self.label_map = build_label_map(np.array(values), include_background=True)
        self.num_classes = len(set(self.label_map.values()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lbl_path, case, week = self.pairs[idx]

        # --- load ---
        vol, _ = read_nifti(img_path, orient=True)     # (D,H,W)
        lab, _ = read_nifti(lbl_path, orient=True)     # (D,H,W)

        # --- normalize image ---
        vol = vol.astype(np.float32, copy=False)
        vol = zscore(vol)

        # --- pad/crop (same center window) ---
        if self.target is not None:
            vol = pad_or_crop_center(vol, self.target, pad_value=0)
            lab = pad_or_crop_center(lab, self.target, pad_value=0)

        # --- to tensors ---
        vol = torch.from_numpy(vol[None, ...])  # (1,D,H,W)

        lab = lab.astype(np.int32, copy=False)
        if self.one_hot:
            lab_oh = to_one_hot(lab, self.label_map, dtype=np.uint8)  # (C,D,H,W)
            lab_t = torch.from_numpy(lab_oh)
        else:
            # map raw -> contiguous
            mapped = np.zeros_like(lab, dtype=np.uint8)
            for raw, k in self.label_map.items():
                mapped[lab == raw] = k
            lab_t = torch.from_numpy(mapped)  # (D,H,W)

        return vol, lab_t


# Expose NUM_CLASSES after you create a dataset instance.
# If you prefer a constant for your model init, you can quick-scan all labels:
def infer_num_classes() -> int:
    ps = find_pairs(IMAGES_DIR, LABELS_DIR)
    vals = scan_label_values(ps, max_files=None)
    lm = build_label_map(np.array(vals), include_background=True)
    return len(set(lm.values()))


NUM_CLASSES = infer_num_classes()
