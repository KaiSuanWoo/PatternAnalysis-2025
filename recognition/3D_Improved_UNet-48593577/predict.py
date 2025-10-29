import os, json, argparse, math, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import find_pairs, split_by_patient, Prostate3DDataset, NUM_CLASSES
from modules import UNet3D

# -----------------------------
# Metrics
# -----------------------------
def dice_per_class(pred_idx, gt_idx, num_classes, eps=1e-6):
    """
    pred_idx, gt_idx: (D,H,W) integer labels [0..C-1]
    returns: np.array shape (C,)
    """
    dices = []
    for c in range(num_classes):
        p = (pred_idx == c)
        g = (gt_idx == c)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        d = (2.0 * inter + eps) / (denom + eps)
        dices.append(float(d))
    return np.array(dices, dtype=np.float32)

def overlay_slice(ax, vol, gt_idx, pr_idx, z):
    """Draw axial slice z with GT and Pred contours."""
    ax.imshow(vol[z], cmap="gray", interpolation="nearest")
    # simple boundaries: show class>0 as a single mask to keep it light
    gt = (gt_idx[z] > 0).astype(np.uint8)
    pr = (pr_idx[z] > 0).astype(np.uint8)
    ax.contour(gt, levels=[0.5], linewidths=1.0)
    ax.contour(pr, levels=[0.5], linewidths=1.0, linestyles="--")
    ax.set_axis_off()

# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(ckpt_path, target, save_dir, max_vis=12, batch_size=1, base_ch=None, depth=None):
    os.makedirs(save_dir, exist_ok=True)

    # ----------------- load checkpoint -----------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # prefer hyperparams from checkpoint for safety
    args_in_ckpt = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    base_ch = base_ch if base_ch is not None else args_in_ckpt.get("base_ch", 32)
    depth = depth if depth is not None else args_in_ckpt.get("depth", 4)
    target = tuple(target) if target is not None else tuple(args_in_ckpt.get("target", [96,192,192]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=num_classes, base_ch=base_ch, depth=depth,
                   use_se=not args_in_ckpt.get("no_se", False),
                   use_att=not args_in_ckpt.get("no_att", False),
                   dropout=args_in_ckpt.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----------------- data (val split) -----------------
    pairs = find_pairs()
    _, val_pairs = split_by_patient(pairs, train_ratio=args_in_ckpt.get("train_ratio", 0.8),
                                    seed=args_in_ckpt.get("seed", 1337))
    val_ds = Prostate3DDataset(val_pairs, target_dhw=target, one_hot=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=2, pin_memory=True)

    # ----------------- loop -----------------
    case_rows = []
    per_class_sum = np.zeros(num_classes, dtype=np.float64)
    n_cases = 0

    vis_count = 0
    vis_grid = min(max_vis, len(val_ds))
    grid_cols = 4
    grid_rows = int(math.ceil(vis_grid / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*3))
    axes = np.array(axes).reshape(grid_rows, grid_cols)

    for i, (imgs, gts) in enumerate(val_loader):
        imgs = imgs.to(device)
        logits = model(imgs)
        pr_idx = torch.argmax(logits, dim=1).cpu().numpy()  # (B,D,H,W)
        gts = gts.cpu().numpy()

        for b in range(pr_idx.shape[0]):
            p = pr_idx[b]
            g = gts[b]
            dices = dice_per_class(p, g, num_classes)
            per_class_sum += dices
            n_cases += 1

            # save row
            mean_d = float(dices.mean())
            case_rows.append([i*batch_size+b, mean_d] + [float(x) for x in dices])

            # qualitative
            if vis_count < vis_grid:
                # take mid axial slice
                vol = imgs[b,0].cpu().numpy()
                z = vol.shape[0] // 2
                r = vis_count // grid_cols
                c = vis_count % grid_cols
                overlay_slice(axes[r, c], vol, g, p, z)
                axes[r, c].set_title(f"Case {i*batch_size+b}  z={z}")
                vis_count += 1

    # tidy empty axes
    for k in range(vis_count, grid_rows*grid_cols):
        r = k // grid_cols; c = k % grid_cols
        axes[r, c].axis("off")

    fig.suptitle("Mid-slice overlays (GT solid, Pred dashed)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overlays_grid.png"), dpi=150)

    # ----------------- summaries -----------------
    mean_per_class = (per_class_sum / max(n_cases,1)).astype(np.float32)
    mean_dice = float(mean_per_class.mean())

    # CSV per-case
    csv_path = os.path.join(save_dir, "per_case_dice.csv")
    header = ["case_idx", "mean_dice"] + [f"class_{k}" for k in range(num_classes)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(case_rows)

    # JSON summary
    summary = {
        "num_classes": int(num_classes),
        "n_cases": int(n_cases),
        "mean_dice": mean_dice,
        "mean_dice_per_class": mean_per_class.tolist(),
        "target_dhw": list(target),
        "checkpoint": os.path.basename(ckpt_path),
        "device": str(device),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    # 1) Per-class bar
    plt.figure(figsize=(6,4))
    xs = np.arange(num_classes)
    plt.bar(xs, mean_per_class)
    plt.xticks(xs, [f"C{k}" for k in xs])
    plt.ylabel("Dice")
    plt.ylim(0,1)
    plt.title("Mean Dice by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_per_class.png"), dpi=150)

    # 2) Histogram of per-case mean dice
    per_case_means = [r[1] for r in case_rows]
    plt.figure(figsize=(6,4))
    plt.hist(per_case_means, bins=10)
    plt.xlabel("Per-case mean Dice")
    plt.ylabel("Count")
    plt.title("Dice distribution (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_hist.png"), dpi=150)

    # Console summary
    print("\n=== Evaluation Summary ===")
    print(f"#cases: {n_cases}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("Per-class Dice:", np.round(mean_per_class, 4))
    print(f"Saved: {save_dir}")
    return summary
# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to *_best.pt")
    ap.add_argument("--save_dir", default=None, help="Folder for results")
    ap.add_argument("--target", type=int, nargs=3, default=None, help="D H W; defaults to checkpoint args")
    ap.add_argument("--base_ch", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    sd = args.save_dir or os.path.join("results", os.path.basename(args.ckpt).replace(".pt","")+"_eval")
    os.makedirs(sd, exist_ok=True)
    evaluate(args.ckpt, args.target, sd, batch_size=args.batch_size)
