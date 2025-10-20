import os, glob, argparse, numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


# ---------- I/O helpers ----------
def _closest_canonical(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nii)

def _load_pair(img_path: str, msk_path: str, orient=True):
    img_nii = nib.load(img_path)
    msk_nii = nib.load(msk_path)
    if orient:
        img_nii = _closest_canonical(img_nii)
        msk_nii = _closest_canonical(msk_nii)
    img = img_nii.get_fdata(caching="unchanged")
    msk = msk_nii.get_fdata(caching="unchanged")
    # collapse possible 4D to 3D
    if img.ndim == 4: img = img[..., 0]
    if msk.ndim == 4: msk = msk[..., 0]
    # binary mask (dataset is prostate vs background)
    msk = (msk > 0).astype(np.uint8)
    # z-score normalise over non-zero voxels
    nz = img[img != 0]
    if nz.size > 0:
        mu, sd = nz.mean(), nz.std() + 1e-8
        img = (img - mu) / sd
    # scale to 0..1 for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img, msk

def _find_cases(root):
    img_dir = os.path.join(root, "semantic_MRs_anon")
    lab_dir = os.path.join(root, "semantic_labels_anon")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    labs = sorted(glob.glob(os.path.join(lab_dir, "*.nii*")))

    def key(p):
        b = os.path.basename(p)
        k = b.split("_LFOV")[0]
        k = k.replace("_SEMANTIC", "")
        return k

    imap = {key(p): p for p in imgs}
    lmap = {key(p): p for p in labs}
    keys = sorted(set(imap) & set(lmap))
    return [(k, imap[k], lmap[k]) for k in keys]


# ---------- viz utils ----------
def _get_plane(volume, plane):
    if plane == "axial":
        return volume  # (Z,Y,X)
    elif plane == "coronal":
        return np.transpose(volume, (1, 0, 2))  # (Y,Z,X)
    elif plane == "sagittal":
        return np.transpose(volume, (2, 0, 1))  # (X,Z,Y)
    else:
        raise ValueError("plane must be one of: axial, coronal, sagittal")

def grid_show(img, msk, plane="axial", ncols=6, nrows=3, alpha=0.4, save_path=None):
    vol = _get_plane(img, plane)
    lbl = _get_plane(msk, plane)
    total = ncols * nrows
    idxs = np.linspace(0, vol.shape[0]-1, total, dtype=int)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.atleast_2d(axes)
    for ax, z in zip(axes.ravel(), idxs):
        ax.imshow(vol[z], cmap="gray")
        ax.imshow(lbl[z], cmap="Reds", alpha=alpha, vmin=0, vmax=1)
        ax.set_title(f"{plane} z={z}")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
        plt.close()
    else:
        plt.show()

def make_gif(img, msk, plane="axial", alpha=0.35, out_path="out.gif", fps=12):
    if not HAS_IMAGEIO:
        print("imageio not installed; `pip install imageio` to enable GIF export.")
        return
    vol = _get_plane(img, plane)
    lbl = _get_plane(msk, plane)
    frames = []
    for z in range(vol.shape[0]):
        base = (vol[z] * 255).astype(np.uint8)
        base_rgb = np.stack([base]*3, axis=-1)
        overlay = (lbl[z] > 0).astype(np.uint8)
        color = np.zeros_like(base_rgb); color[..., 0] = 255
        blended = (1-alpha)*base_rgb + alpha*(overlay[..., None]*color)
        frames.append(blended.astype(np.uint8))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[saved] {out_path}")

# ---------- interactive viewer ----------
class ScrollViewer:
    def __init__(self, img, msk, plane="axial", alpha=0.4):
        self.img_full = img; self.msk_full = msk
        self.alpha = alpha
        self.set_plane(plane)
        self.z = self.vol.shape[0] // 2
        self.overlay = True
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update()

    def set_plane(self, plane):
        self.plane = plane
        self.vol = _get_plane(self.img_full, plane)
        self.lbl = _get_plane(self.msk_full, plane)

    def update(self):
        self.ax.clear()
        self.ax.imshow(self.vol[self.z], cmap="gray")
        if self.overlay:
            self.ax.imshow(self.lbl[self.z], cmap="Reds", alpha=self.alpha, vmin=0, vmax=1)
        self.ax.set_title(f"{self.plane} z={self.z} | overlay={'on' if self.overlay else 'off'}")
        self.ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        if e.key in ["up", "right"]:
            self.z = min(self.z + 1, self.vol.shape[0]-1)
        elif e.key in ["down", "left"]:
            self.z = max(self.z - 1, 0)
        elif e.key == "o":
            self.overlay = not self.overlay
        elif e.key == "a":
            self.set_plane("axial");   self.z = self.vol.shape[0]//2
        elif e.key == "s":
            self.set_plane("sagittal"); self.z = self.vol.shape[0]//2
        elif e.key == "d":
            self.set_plane("coronal");  self.z = self.vol.shape[0]//2
        self.update()

    def show(self):
        print("Keys: ←/→/↑/↓ scroll • o toggle overlay • a axial • s sagittal • d coronal • q to quit")
        plt.show()


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("Visualise Prostate3D MRI volumes and labels")
    ap.add_argument("--data_root", default="Prostate3D_data")
    ap.add_argument("--pid", default=None, help="Case id like 'Case_033_Week0'. If omitted, list cases.")
    ap.add_argument("--mode", choices=["grid", "interactive", "gif"], default="grid")
    ap.add_argument("--plane", choices=["axial","coronal","sagittal"], default="axial")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--save", type=str, default=None, help="Path to save grid/gif (auto created).")
    return ap.parse_args()

def main():
    args = parse_args()
    cases = _find_cases(args.data_root)
    if not cases:
        raise SystemExit("No cases found. Check folder structure.")
    if args.pid is None:
        print("Available cases (sample):")
        for k, _, _ in cases[:20]:
            print("  ", k)
        print("Use: --pid <Case_XXX_WeekY>")
        return

    match = [c for c in cases if c[0] == args.pid]
    if not match:
        raise SystemExit(f"Case id '{args.pid}' not found.")
    pid, img_path, msk_path = match[0]
    print(f"[info] Loading {pid}")
    img, msk = _load_pair(img_path, msk_path, orient=True)

    if args.mode == "grid":
        out = args.save or os.path.join("runs", f"{pid}_{args.plane}_grid.png")
        grid_show(img, msk, plane=args.plane, nrows=args.rows, ncols=args.cols, alpha=args.alpha, save_path=out)
    elif args.mode == "gif":
        out = args.save or os.path.join("runs", f"{pid}_{args.plane}.gif")
        make_gif(img, msk, plane=args.plane, alpha=args.alpha, out_path=out, fps=10)
    else:
        ScrollViewer(img, msk, plane=args.plane, alpha=args.alpha).show()

if __name__ == "__main__":
    main()
