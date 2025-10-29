# dataset.py
import os, re, glob, random
from typing import Dict, List, Sequence, Tuple
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ----------- CONFIG: set your data roots here -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("PROSTATE3D_DATA", os.path.join(_HERE, "Prostate3D_data"))
MR_DIR  = os.path.join(_DATA_ROOT, "semantic_MRs_anon")
LAB_DIR = os.path.join(_DATA_ROOT, "semantic_labels_anon")

# Number of classes incl. background. Adjust if your labels differ.
NUM_CLASSES = 6  # [0..5] -> 0=bg, 1..5 organs (body, bone, bladder, rectum, prostate)

# If your labels are not exactly {0..5}, remap here, e.g., {0:0, 10:1, 20:2, ...}
REMAP: Dict[int, int] = {}  # leave empty for identity


# ----------- FILE ID HELPERS -----------
_ID_MR  = re.compile(r"(Case_\d+_Week\d+)_LFOV\.nii\.gz$")
_ID_LAB = re.compile(r"(Case_\d+_Week\d+)_SEMANTIC_LFOV\.nii\.gz$")

def _mr_id(path: str) -> str:
    m = _ID_MR.match(os.path.basename(path))
    if not m: raise ValueError(f"Unexpected MR filename: {path}")
    return m.group(1)

def _lab_id(path: str) -> str:
    m = _ID_LAB.match(os.path.basename(path))
    if not m: raise ValueError(f"Unexpected label filename: {path}")
    return m.group(1)

def _patient_id_from_case(case_week: str) -> str:
    # "Case_004_Week1" -> "Case_004"
    return re.match(r"(Case_\d+)_Week\d+", case_week).group(1)


# ----------- PAIR DISCOVERY + CHECKS -----------
def find_pairs(mr_dir: str = MR_DIR, lab_dir: str = LAB_DIR) -> List[Tuple[str, str]]:
    mrs  = { _mr_id(p): p  for p in glob.glob(os.path.join(mr_dir,  "*.nii.gz")) }
    labs = { _lab_id(p): p for p in glob.glob(os.path.join(lab_dir, "*.nii.gz")) }
    ids = sorted(set(mrs) & set(labs))
    pairs = [(mrs[i], labs[i]) for i in ids]
    if not pairs:
        raise RuntimeError(f"No MR/label pairs found. MR_DIR={mr_dir} LAB_DIR={lab_dir}")
    return pairs

def remap_labels(arr: np.ndarray) -> np.ndarray:
    if not REMAP:  # identity
        return arr
    vec = np.vectorize(lambda v: REMAP.get(int(v), int(v)))
    return vec(arr).astype(np.uint8)

def sanity_check(pairs: Sequence[Tuple[str,str]], check_affine=True) -> Dict[str, object]:
    mismatches = []
    classes = set()
    shapes = set()
    for mr_p, lab_p in pairs:
        img = nib.load(mr_p)
        lab = nib.load(lab_p)
        if img.shape != lab.shape:
            mismatches.append((mr_p, lab_p, "shape", img.shape, lab.shape))
            continue
        if check_affine and not np.allclose(img.affine, lab.affine, atol=1e-3):
            mismatches.append((mr_p, lab_p, "affine"))
        lab_np = remap_labels(lab.get_fdata().astype(np.float32))
        classes.update(np.unique(lab_np).astype(int).tolist())
        shapes.add(img.shape)
    return {
        "num_pairs": len(pairs),
        "classes": sorted(classes),
        "unique_shapes": sorted(list(shapes)),
        "mismatches": mismatches,
    }

def split_by_patient(
    pairs: Sequence[Tuple[str,str]],
    train_ratio=0.7, val_ratio=0.15, seed=1337
) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]], List[Tuple[str,str]]]:
    # group all weeks from the same Case_xxx together
    from collections import defaultdict
    groups = defaultdict(list)
    for mr_p, lab_p in pairs:
        cid = _patient_id_from_case(_mr_id(mr_p))
        groups[cid].append((mr_p, lab_p))
    pids = list(groups.keys())
    random.Random(seed).shuffle(pids)
    n = len(pids)
    n_tr = int(train_ratio * n)
    n_va = int(val_ratio * n)
    tr_ids = pids[:n_tr]
    va_ids = pids[n_tr:n_tr+n_va]
    te_ids = pids[n_tr+n_va:]
    train_pairs = sum((groups[i] for i in tr_ids), [])
    val_pairs   = sum((groups[i] for i in va_ids), [])
    test_pairs  = sum((groups[i] for i in te_ids), [])
    return train_pairs, val_pairs, test_pairs


# ----------- DATASET (patch sampling + body-masked z-score) -----------
def _zscore_in_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = vol[mask > 0]
    if m.size == 0:
        return (vol - vol.mean()) / (vol.std() + 1e-6)
    return (vol - m.mean()) / (m.std() + 1e-6)

def _bbox_from_mask(mask: np.ndarray, pad: int = 8):
    idx = np.where(mask > 0)
    if idx[0].size == 0:
        D,H,W = mask.shape
        return (slice(0,D), slice(0,H), slice(0,W))
    zmin, zmax = max(idx[0].min() - pad, 0), min(idx[0].max() + pad, mask.shape[0])
    ymin, ymax = max(idx[1].min() - pad, 0), min(idx[1].max() + pad, mask.shape[1])
    xmin, xmax = max(idx[2].min() - pad, 0), min(idx[2].max() + pad, mask.shape[2])
    return (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))

class Prostate3DDataset(Dataset):
    """
    Yields (img, lab) with shapes (1, D, H, W) and (D, H, W).
    - Normalizes inside body mask (labels>0)
    - Crops to body bbox (speed)
    - Samples 3D patches if patch_size is not None
    """
    def __init__(
        self,
        pairs: Sequence[Tuple[str,str]],
        patch_size: Tuple[int,int,int] = (96,96,96),
        augment: bool = False,
        patch_prob_fg: float = 0.5,
    ):
        self.pairs = list(pairs)
        self.patch = patch_size
        self.aug = augment
        self.pfg = patch_prob_fg

    def __len__(self): return len(self.pairs)

    def _load_case(self, mr_p: str, lab_p: str) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(mr_p).get_fdata().astype(np.float32)
        lab = remap_labels(nib.load(lab_p).get_fdata().astype(np.float32))
        body = (lab > 0).astype(np.uint8)
        img = _zscore_in_mask(img, body)
        z,y,x = _bbox_from_mask(body, pad=8)
        return img[z, y, x], lab[z, y, x]

    def _sample_patch(self, img: np.ndarray, lab: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        if self.patch is None:
            return img, lab
        D,H,W = img.shape
        pd,ph,pw = self.patch
        # choose center
        if np.random.rand() < self.pfg and lab.sum() > 0:
            zs,ys,xs = np.where(lab > 0)
            i = np.random.randint(0, len(zs))
            cz,cy,cx = int(zs[i]), int(ys[i]), int(xs[i])
        else:
            cz = np.random.randint(pd//2, max(pd//2+1, D-pd//2))
            cy = np.random.randint(ph//2, max(ph//2+1, H-ph//2))
            cx = np.random.randint(pw//2, max(pw//2+1, W-pw//2))
        z0,y0,x0 = np.clip([cz - pd//2, cy - ph//2, cx - pw//2], 0, None)
        z1,y1,x1 = min(z0+pd, D), min(y0+ph, H), min(x0+pw, W)
        z0,y0,x0 = z1-pd, y1-ph, x1-pw
        return img[z0:z1, y0:y1, x0:x1], lab[z0:z1, y0:y1, x0:x1]

    def __getitem__(self, idx: int):
        mr_p, lab_p = self.pairs[idx]
        img, lab = self._load_case(mr_p, lab_p)
        img, lab = self._sample_patch(img, lab)

        if self.aug:
            # simple, robust 3D flips
            if np.random.rand() < 0.5:
                img, lab = img[::-1].copy(), lab[::-1].copy()
            if np.random.rand() < 0.5:
                img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, lab = img[:, :, ::-1].copy(), lab[:, :, ::-1].copy()


        # to tensors
        img_t = torch.from_numpy(img[None])              # (1, D, H, W)
        lab_t = torch.from_numpy(lab.astype(np.int64))   # (D, H, W)
        return img_t, lab_t


# ----------- OPTIONAL: quick overlay for QC -----------
def show_overlay(mr_path: str, lab_path: str, zs=range(30, 256, 15), alpha=0.35):
    import matplotlib.pyplot as plt
    img = nib.load(mr_path).get_fdata().astype(np.float32)
    lab = remap_labels(nib.load(lab_path).get_fdata().astype(np.float32))
    body = (lab > 0).astype(np.uint8)
    img = _zscore_in_mask(img, body)

    zs = [z for z in zs if z < img.shape[2]]
    cols = 5; rows = (len(zs)+cols-1)//cols
    import numpy.ma as ma
    plt.figure(figsize=(3.6*cols, 3.6*rows))
    for k, z in enumerate(zs):
        ax = plt.subplot(rows, cols, k+1)
        ax.imshow(img[:,:,z].T, cmap="gray", origin="lower")
        ax.imshow(ma.masked_where(lab[:,:,z].T==0, lab[:,:,z].T), alpha=alpha)
        ax.set_title(f"axial z={z}"); ax.axis("off")
    plt.tight_layout(); plt.show()


# ----------- SELF-TEST (run `python dataset.py`) -----------
if __name__ == "__main__":
    pairs = find_pairs(MR_DIR, LAB_DIR)
    info = sanity_check(pairs)
    print(f"Pairs: {info['num_pairs']}")
    print(f"Classes found: {info['classes']}")
    print(f"Unique shapes: {info['unique_shapes']}")
    if info["mismatches"]:
        print("Mismatches:", info["mismatches"][:3], "...")

    tr, va, te = split_by_patient(pairs, train_ratio=0.7, val_ratio=0.15, seed=1337)
    print(f"Split -> train {len(tr)} | val {len(va)} | test {len(te)}")

    # Quick peek overlay (comment out if running headless)
    # show_overlay(tr[0][0], tr_
