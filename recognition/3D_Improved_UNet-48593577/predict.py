# predict.py — full-volume inference + per-case Dice + overlays
import os, json, math, numpy as np, torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Tuple, List

from modules import ImprovedUNet3D, UNet3D
from dataset import find_patients, load_data_3D   # uses your updated loader
from utils import check_cuda, set_seed, dice_coefficient

# --- CONFIG (edit) ---
CFG = {
    "DATA_ROOT": "Prostate3D_data",
    "SAVE_DIR":  "results",
    "CKPT":      "runs/best.ckpt",
    "MODEL":     "improved",       # "improved" or "unet"
    "PATCH":     (96,96,96),       # can be (64,64,64) on CPU
    "OVERLAP":   0.5,              # 50% overlap
    "THRESH":    0.5,              # binarization
    "SEED":      1337,
    # Split: use last 15% as test (train.py used 70/30 → here we carve 15% from the tail)
    "TEST_FRACTION": 0.15,
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _sliding_window_pred(model, vol: torch.Tensor, patch: Tuple[int,int,int], overlap: float, device) -> torch.Tensor:
    """vol: (1,1,D,H,W) float32; returns prob map (1,1,D,H,W)."""
    model.eval()
    with torch.no_grad():
        _, _, D, H, W = vol.shape
        pd, ph, pw = patch
        sd = max(1, int(pd * (1 - overlap)))
        sh = max(1, int(ph * (1 - overlap)))
        sw = max(1, int(pw * (1 - overlap)))

        prob = torch.zeros_like(vol)
        cnt  = torch.zeros_like(vol)

        for z in range(0, max(1, D - pd + 1), sd):
            for y in range(0, max(1, H - ph + 1), sh):
                for x in range(0, max(1, W - pw + 1), sw):
                    patch_vol = vol[:, :, z:z+pd, y:y+ph, x:x+pw]
                    # pad borders if needed
                    pz = pd - patch_vol.shape[2]; py = ph - patch_vol.shape[3]; px = pw - patch_vol.shape[4]
                    if pz>0 or py>0 or px>0:
                        patch_vol = F.pad(patch_vol, (0,px, 0,py, 0,pz))
                    pr = model(patch_vol.to(device))  # (1,1,pd,ph,pw)
                    pr = pr[:, :, :patch_vol.shape[2]-pz if pz>0 else pd,
                                :patch_vol.shape[3]-py if py>0 else ph,
                                :patch_vol.shape[4]-px if px>0 else pw]
                    prob[:, :, z:z+pr.shape[2], y:y+pr.shape[3], x:x+pr.shape[4]] += pr.cpu()
                    cnt[:, :,  z:z+pr.shape[2], y:y+pr.shape[3], x:x+pr.shape[4]]  += 1
        cnt[cnt==0] = 1
        return prob / cnt

def _central_overlay(img, gt, pr, save_path):
    """Save central axial overlays (image, GT, Pred)."""
    ensure_dir(os.path.dirname(save_path))
    img = img.squeeze(0)  # (D,H,W)
    gt  = gt.squeeze(0); pr = pr.squeeze(0)
    z = img.shape[0]//2
    img2 = (img[z]-img.min())/(img.max()-img.min()+1e-6)
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(img2, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(gt[z], cmap="gray"); axs[1].set_title("GT"); axs[1].axis("off")
    axs[2].imshow(pr[z], cmap="gray"); axs[2].set_title("Pred"); axs[2].axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def main():
    cfg = CFG
    device = check_cuda()
    set_seed(cfg["SEED"])
    ensure_dir(cfg["SAVE_DIR"])

    # IDs and test split
    items = find_patients(cfg["DATA_ROOT"])
    ids = [it.pid for it in items]
    n = len(ids); n_test = max(1, int(n * cfg["TEST_FRACTION"]))
    test_ids = ids[-n_test:]  # tail as test set (patient-wise)

    # Model
    model = ImprovedUNet3D(1,1,32) if cfg["MODEL"]=="improved" else UNet3D(1,1,32)
    sd = torch.load(cfg["CKPT"], map_location="cpu")
    model.load_state_dict(sd); model.to(device).eval()
    print(f"[info] Loaded {cfg['MODEL']} from {cfg['CKPT']} | test cases={len(test_ids)}")

    # Evaluate each case
    results = []
    for pid in test_ids:
        item = next(it for it in items if it.pid==pid)
        img = load_data_3D([item.img_path], normImage=True, orient=True, label_mode=False)[0]   # (D,H,W) float32
        msk = load_data_3D([item.msk_path], normImage=False, orient=True, label_mode=True)[0]   # (D,H,W) uint8
        msk = (msk > 0).astype(np.float32)

        vol = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
        prob = _sliding_window_pred(model, vol, cfg["PATCH"], cfg["OVERLAP"], device)          # (1,1,D,H,W)
        pred = (prob >= cfg["THRESH"]).float()

        gt_t  = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).float()
        dice = float(dice_coefficient(pred, gt_t))

        # save overlay
        _central_overlay(vol[0,0].cpu().numpy()[None,...],
                         gt_t[0,0].cpu().numpy()[None,...],
                         pred[0,0].cpu().numpy()[None,...],
                         os.path.join(cfg["SAVE_DIR"], f"{pid}_overlay.png"))

        results.append({"pid": pid, "dice": round(dice, 4)})

    # summary
    mean_dice = float(np.mean([r["dice"] for r in results])) if results else 0.0
    with open(os.path.join(cfg["SAVE_DIR"], "summary.json"), "w") as f:
        json.dump({"mean_dice": round(mean_dice,4), "cases": results}, f, indent=2)
    print(f"✅ Test mean Dice = {mean_dice:.4f}  | results → {cfg['SAVE_DIR']}")

if __name__ == "__main__":
    main()
