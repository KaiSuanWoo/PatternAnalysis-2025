import os, json, argparse, math, csv, re
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import find_pairs, split_by_patient, Prostate3DDataset, NUM_CLASSES
from modules import UNet3D

# -------------- helpers: logs --------------
def _derive_log_path_from_ckpt(ckpt_path: str) -> str:
    # ckpt looks like: runs/<runname>_best.pt  ->  runs/<runname>_log.jsonl
    base = os.path.basename(ckpt_path)
    log_base = re.sub(r"_best\.pt$", "_log.jsonl", base)
    return os.path.join(os.path.dirname(ckpt_path), log_base)

def _load_training_log(log_path: str):
    if not os.path.exists(log_path):
        return None
    epochs, tr_loss, tr_dice, val_loss, val_dice = [], [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            epochs.append(row.get("epoch"))
            tr_loss.append(row.get("train_loss"))
            tr_dice.append(row.get("train_dice"))
            val_loss.append(row.get("val_loss"))
            val_dice.append(row.get("val_dice"))
    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(tr_loss),
        "train_dice": np.array(tr_dice),
        "val_loss": np.array(val_loss),
        "val_dice": np.array(val_dice),
    }

def _plot_training_curves(log, save_dir):
    # Dice vs epoch
    plt.figure(figsize=(6,4))
    if log is not None:
        plt.plot(log["epoch"], log["train_dice"], label="Train Dice")
        plt.plot(log["epoch"], log["val_dice"],   label="Val Dice")
        plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.title("Dice vs Epoch"); plt.legend()
    else:
        plt.text(0.5, 0.5, "No log file found", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "dice_vs_epoch.png"), dpi=150)

    # Loss vs epoch
    plt.figure(figsize=(6,4))
    if log is not None:
        plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
        plt.plot(log["epoch"], log["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend()
    else:
        plt.text(0.5, 0.5, "No log file found", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"), dpi=150)

# -------------- metrics & viz --------------
def dice_per_class(pred_idx, gt_idx, num_classes, eps=1e-6):
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
    ax.imshow(vol[z], cmap="gray", interpolation="nearest")
    gt = (gt_idx[z] > 0).astype(np.uint8)
    pr = (pr_idx[z] > 0).astype(np.uint8)
    ax.contour(gt, levels=[0.5], linewidths=1.0)
    ax.contour(pr, levels=[0.5], linewidths=1.0, linestyles="--")
    ax.set_axis_off()

def save_triplet(vol, gt_idx, pr_idx, z, out_path):
    """Save a 1x3 panel: Original | Label | Prediction at axial slice z."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(vol[z], cmap="gray", interpolation="nearest")
    axs[0].set_title("Original"); axs[0].axis("off")

    im1 = axs[1].imshow(gt_idx[z], interpolation="nearest")
    axs[1].set_title("Label"); axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(pr_idx[z], interpolation="nearest")
    axs[2].set_title("Prediction"); axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -------------- evaluation --------------
@torch.no_grad()
def evaluate(ckpt_path, target, save_dir, max_vis=12, max_triplets=12, batch_size=1,
             base_ch=None, depth=None):
    os.makedirs(save_dir, exist_ok=True)

    # Load ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_in_ckpt = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    base_ch = base_ch if base_ch is not None else args_in_ckpt.get("base_ch", 32)
    depth   = depth   if depth   is not None else args_in_ckpt.get("depth", 4)
    target  = tuple(target) if target is not None else tuple(args_in_ckpt.get("target", [96,192,192]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_ch=base_ch,
        depth=depth,
        use_se=not args_in_ckpt.get("no_se", False),
        use_att=not args_in_ckpt.get("no_att", False),
        dropout=args_in_ckpt.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Data (val split)
    pairs = find_pairs()
    _, val_pairs = split_by_patient(pairs,
                                    train_ratio=args_in_ckpt.get("train_ratio", 0.8),
                                    seed=args_in_ckpt.get("seed", 1337))
    val_ds = Prostate3DDataset(val_pairs, target_dhw=target, one_hot=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=2, pin_memory=True)

    # Training curves
    log_path = _derive_log_path_from_ckpt(ckpt_path)
    log = _load_training_log(log_path)
    _plot_training_curves(log, save_dir)

    # Eval loop
    case_rows = []
    per_class_sum = np.zeros(num_classes, dtype=np.float64)
    n_cases = 0

    # overlays grid
    vis_grid = min(max_vis, len(val_ds))
    grid_cols = 4
    grid_rows = int(math.ceil(vis_grid / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*3))
    axes = np.array(axes).reshape(grid_rows, grid_cols)
    vis_count = 0

    triplet_count = 0

    for i, (imgs, gts) in enumerate(val_loader):
        imgs = imgs.to(device)
        logits = model(imgs)
        pr_idx = torch.argmax(logits, dim=1).cpu().numpy()
        gts = gts.cpu().numpy()

        for b in range(pr_idx.shape[0]):
            p = pr_idx[b]
            g = gts[b]
            vol = imgs[b, 0].cpu().numpy()

            dices = dice_per_class(p, g, num_classes)
            per_class_sum += dices
            n_cases += 1

            mean_d = float(dices.mean())
            case_rows.append([i*batch_size+b, mean_d] + [float(x) for x in dices])

            # overlays grid (mid slice)
            if vis_count < vis_grid:
                z = vol.shape[0] // 2
                r = vis_count // grid_cols
                c = vis_count % grid_cols
                overlay_slice(axes[r, c], vol, g, p, z)
                axes[r, c].set_title(f"Case {i*batch_size+b}  z={z}")
                vis_count += 1

            # triplet side-by-side
            if triplet_count < max_triplets:
                z = vol.shape[0] // 2
                out_path = os.path.join(save_dir, f"case_{i*batch_size+b:03d}_triplet_z{z}.png")
                save_triplet(vol, g, p, z, out_path)
                triplet_count += 1

    # tidy empty overlay axes
    for k in range(vis_count, grid_rows*grid_cols):
        r = k // grid_cols; c = k % grid_cols
        axes[r, c].axis("off")
    fig.suptitle("Mid-slice overlays (GT solid, Pred dashed)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "overlays_grid.png"), dpi=150)

    # Summaries
    mean_per_class = (per_class_sum / max(n_cases,1)).astype(np.float32)
    mean_dice = float(mean_per_class.mean())

    # per-case CSV
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

    # Per-class bar
    plt.figure(figsize=(6,4))
    xs = np.arange(num_classes)
    plt.bar(xs, mean_per_class)
    plt.xticks(xs, [f"C{k}" for k in xs])
    plt.ylabel("Dice"); plt.ylim(0,1); plt.title("Mean Dice by Class")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "dice_per_class.png"), dpi=150)

    # Per-case mean histogram
    per_case_means = [r[1] for r in case_rows]
    plt.figure(figsize=(6,4))
    plt.hist(per_case_means, bins=10)
    plt.xlabel("Per-case mean Dice"); plt.ylabel("Count"); plt.title("Dice distribution (val)")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "dice_hist.png"), dpi=150)

    # Console
    print("\n=== Evaluation Summary ===")
    print(f"#cases: {n_cases}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("Per-class Dice:", np.round(mean_per_class, 4))
    print(f"Saved: {save_dir}")
    return summary

# -------------- CLI --------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to *_best.pt")
    ap.add_argument("--save_dir", default=None, help="Folder for results")
    ap.add_argument("--target", type=int, nargs=3, default=None, help="D H W; defaults to checkpoint args")
    ap.add_argument("--base_ch", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_triplets", type=int, default=12, help="How many side-by-side panels to save")
    args = ap.parse_args()

    sd = args.save_dir or os.path.join("results", os.path.basename(args.ckpt).replace(".pt","")+"_eval")
    os.makedirs(sd, exist_ok=True)
    evaluate(args.ckpt, args.target, sd, batch_size=args.batch_size, max_triplets=args.max_triplets)
