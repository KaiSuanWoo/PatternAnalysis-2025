import os, json, math, re
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import find_pairs, split_by_patient, Prostate3DDataset, NUM_CLASSES
from modules import UNet3D


# --------------------- utility: latest checkpoint ---------------------
def find_latest_ckpt(runs_dir="runs"):
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"No runs directory found at {runs_dir}")
    ckpts = [
        os.path.join(runs_dir, f)
        for f in os.listdir(runs_dir)
        if f.endswith("_best.pt")
    ]
    if not ckpts:
        raise FileNotFoundError(f"No *_best.pt checkpoints in {runs_dir}")
    latest = max(ckpts, key=os.path.getmtime)
    print(f"[predict] Using latest checkpoint: {latest}")
    return latest


def derive_log_path_from_ckpt(ckpt_path: str):
    base = os.path.basename(ckpt_path)
    log_base = re.sub(r"_best\.pt$", "_log.jsonl", base)
    log_path = os.path.join(os.path.dirname(ckpt_path), log_base)
    return log_path if os.path.exists(log_path) else None


# --------------------- training log loading + curve plotting ---------------------
def load_training_log(log_path: str):
    if log_path is None or not os.path.exists(log_path):
        return None

    epochs, train_loss, val_loss, train_dice, val_dice = [], [], [], [], []
    train_dice_per_class, val_dice_per_class = [], []

    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            epochs.append(row.get("epoch"))
            train_loss.append(row.get("train_loss"))
            val_loss.append(row.get("val_loss"))
            train_dice.append(row.get("train_dice"))
            val_dice.append(row.get("val_dice"))
            train_dice_per_class.append(row.get("train_dice_classes"))
            val_dice_per_class.append(row.get("val_dice_classes"))

    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "train_dice": np.array(train_dice),
        "val_dice": np.array(val_dice),
        "train_dice_per_class": train_dice_per_class,
        "val_dice_per_class": val_dice_per_class,
    }


def plot_training_curves(log, save_dir, class_names=None):
    """Plots training/validation loss and per-class dice vs epoch."""
    # ----- Loss -----
    plt.figure(figsize=(6, 4))
    if log is not None and len(log["epoch"]) > 0:
        plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
        plt.plot(log["epoch"], log["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No log data", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"), dpi=150)
    plt.close()

    # ----- Per-class Dice -----
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    if log is not None and len(log["epoch"]) > 0 and any(log["train_dice_per_class"]):
        epochs = log["epoch"]
        train_list = [x for x in log["train_dice_per_class"] if x is not None]
        val_list = [x for x in log["val_dice_per_class"] if x is not None]
        train_arr = np.array(train_list) if len(train_list) > 0 else None
        val_arr = np.array(val_list) if len(val_list) > 0 else None

        # Train subplot
        ax0 = axs[0]
        ax0.set_title("Training Dice per Class")
        if train_arr is not None:
            num_classes = train_arr.shape[1]
            for c in range(num_classes):
                label = class_names[c] if (class_names and c < len(class_names)) else f"Class {c}"
                ax0.plot(epochs[:train_arr.shape[0]], train_arr[:, c], label=label)
            ax0.set_xlabel("Epoch"); ax0.set_ylabel("Dice Coefficient"); ax0.legend()
        else:
            ax0.text(0.5, 0.5, "No per-class train dice", ha="center", va="center")
            ax0.axis("off")

        # Val subplot
        ax1 = axs[1]
        ax1.set_title("Validation Dice per Class")
        if val_arr is not None:
            num_classes = val_arr.shape[1]
            for c in range(num_classes):
                label = class_names[c] if (class_names and c < len(class_names)) else f"Class {c}"
                ax1.plot(epochs[:val_arr.shape[0]], val_arr[:, c], label=label)
            ax1.set_xlabel("Epoch"); ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No per-class val dice", ha="center", va="center")
            ax1.axis("off")
    else:
        for ax in axs:
            ax.text(0.5, 0.5, "No log data", ha="center", va="center")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "dice_vs_epoch_by_class.png"), dpi=150)
    plt.close(fig)


# --------------------- core metrics ---------------------
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


# --------------------- visualization helpers ---------------------
def save_triplet(vol, gt_idx, pr_idx, z, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(vol[z], cmap="gray"); axs[0].set_title("Original"); axs[0].axis("off")
    im1 = axs[1].imshow(gt_idx[z]); axs[1].set_title("Label"); axs[1].axis("off"); plt.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(pr_idx[z]); axs[2].set_title("Prediction"); axs[2].axis("off"); plt.colorbar(im2, ax=axs[2])
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------- main evaluation ---------------------
@torch.no_grad()
def evaluate_checkpoint(ckpt_path, save_dir, batch_size=1, max_triplets=12):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load model ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    target = tuple(args.get("target", [96, 192, 192]))
    base_ch = args.get("base_ch", 32)
    depth = args.get("depth", 4)
    use_se = not args.get("no_se", False)
    use_att = not args.get("no_att", False)
    dropout = args.get("dropout", 0.0)
    train_ratio = args.get("train_ratio", 0.8)
    seed = args.get("seed", 1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(1, num_classes, base_ch, depth, use_se, use_att, dropout).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # ---- Build test loader ----
    all_pairs = find_pairs()
    train_pairs, val_pairs = split_by_patient(all_pairs, train_ratio=train_ratio, seed=seed)
    test_ds = Prostate3DDataset(val_pairs, target_dhw=target, one_hot=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---- Load logs and plot curves ----
    class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
    log_path = derive_log_path_from_ckpt(ckpt_path)
    log_data = load_training_log(log_path)
    plot_training_curves(log_data, save_dir, class_names=class_names)

    # ---- Evaluate ----
    all_dice = []
    saved_triplets, case_idx = 0, 0
    for imgs, gts in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        preds_idx = torch.argmax(preds, dim=1).cpu().numpy()
        gts_np = gts.cpu().numpy()
        for b in range(preds_idx.shape[0]):
            p, g = preds_idx[b], gts_np[b]
            vol = imgs[b, 0].cpu().numpy()
            all_dice.append(dice_per_class(p, g, num_classes))
            if saved_triplets < max_triplets:
                z = vol.shape[0] // 2
                save_triplet(vol, g, p, z, os.path.join(save_dir, f"case_{case_idx:03d}_triplet_z{z}.png"))
                saved_triplets += 1
            case_idx += 1

    all_dice = np.stack(all_dice, axis=0)
    mean_per_class = all_dice.mean(axis=0)
    mean_dice = mean_per_class.mean()
    n_cases = all_dice.shape[0]

    # ---- Plot per-class Dice (TEST) ----
    plt.figure(figsize=(8, 5))
    xs = np.arange(num_classes)
    plt.bar(xs, mean_per_class, color="skyblue", edgecolor="black")
    plt.axhline(mean_dice, color="red", linestyle="--", label=f"Mean Dice = {mean_dice:.3f}")
    plt.xticks(xs, class_names[:num_classes], rotation=15)
    plt.ylabel("Dice Coefficient")
    plt.ylim(0, 1.05)
    plt.title("Per-Class Dice on Test Set")
    plt.legend()
    for i, v in enumerate(mean_per_class):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_per_class_dice.png"), dpi=150)
    plt.close()

    # ---- Plot complete Dice loss over testing ----
    dice_loss = 1.0 - all_dice
    total_loss = 1.0 - all_dice.mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss, label="Total Loss")
    for c in range(num_classes):
        plt.plot(dice_loss[:, c], label=f"{class_names[c]} Loss" if c < len(class_names) else f"Class {c} Loss")
    plt.xlabel("Case Index")
    plt.ylabel("Dice Loss (1 - Dice)")
    plt.title("Complete DICE Loss over Testing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_loss_over_cases.png"), dpi=150)
    plt.close()

    # ---- Save summary ----
    summary = {
        "checkpoint": os.path.basename(ckpt_path),
        "n_cases": int(n_cases),
        "target_dhw": list(target),
        "mean_dice_overall": float(mean_dice),
        "mean_dice_per_class": mean_per_class.tolist(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print results ----
    print("\n=== Evaluation Summary ===")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"#cases: {n_cases}")
    print(f"Mean Dice (overall): {mean_dice:.4f}")
    for cname, val in zip(class_names, mean_per_class):
        print(f"  {cname:10s}: {val:.4f}")
    print(f"Saved outputs to: {save_dir}")


# --------------------- entrypoint ---------------------
if __name__ == "__main__":
    ckpt_path = find_latest_ckpt("runs")
    tag = os.path.basename(ckpt_path).replace("_best.pt", "_eval")
    save_dir = os.path.join("results", tag)

    evaluate_checkpoint(
        ckpt_path=ckpt_path,
        save_dir=save_dir,
        batch_size=1,
        max_triplets=12,
    )
