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

    epochs = []
    train_loss = []
    val_loss = []
    train_dice = []           # mean dice
    val_dice = []             # mean dice
    train_dice_per_class = [] # list of arrays per epoch
    val_dice_per_class = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            epochs.append(row.get("epoch"))
            train_loss.append(row.get("train_loss"))
            val_loss.append(row.get("val_loss"))
            train_dice.append(row.get("train_dice"))
            val_dice.append(row.get("val_dice"))

            # we will allow optional per-class dice if you logged them
            # expecting keys "train_dice_classes": [c0,c1,...], "val_dice_classes": [...]
            tdc = row.get("train_dice_classes")
            vdc = row.get("val_dice_classes")
            train_dice_per_class.append(tdc if tdc is not None else None)
            val_dice_per_class.append(vdc if vdc is not None else None)

    data = {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "train_dice": np.array(train_dice),
        "val_dice": np.array(val_dice),
        "train_dice_per_class": train_dice_per_class,
        "val_dice_per_class": val_dice_per_class,
    }
    return data


def plot_training_curves(log, save_dir, class_names=None):
    """
    Saves:
      - dice_vs_epoch_by_class.png  (2 subplots: train and val per-class Dice curves)
      - loss_vs_epoch.png          (train vs val loss)
    Assumes log["train_dice_per_class"][i] is a list [c0,c1,...] for epoch i.
    """

    # ----- loss vs epoch -----
    plt.figure(figsize=(6,4))
    if log is not None and len(log["epoch"]) > 0:
        plt.plot(log["epoch"], log["train_loss"], label="Training Loss")
        plt.plot(log["epoch"], log["val_loss"],   label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No log data", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"), dpi=150)
    plt.close()

    # ----- dice per class (train & val) -----
    # We'll create a side-by-side figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12,4), sharey=True)
    if log is not None and len(log["epoch"]) > 0 and any(log["train_dice_per_class"]):
        epochs = log["epoch"]

        # stack per-class dice across epochs into arrays shape (num_epochs, num_classes)
        # handle missing per-class entries safely
        train_list = [x for x in log["train_dice_per_class"] if x is not None]
        val_list   = [x for x in log["val_dice_per_class"] if x is not None]

        if len(train_list) > 0:
            train_arr = np.array(train_list)  # (E, C)
        else:
            train_arr = None
        if len(val_list) > 0:
            val_arr = np.array(val_list)      # (E, C)
        else:
            val_arr = None

        # left subplot: training per-class dice
        ax0 = axs[0]
        ax0.set_title("Training Dice Similarity Coefficients For Each Class")
        if train_arr is not None:
            num_classes = train_arr.shape[1]
            for c in range(num_classes):
                label = class_names[c] if (class_names and c < len(class_names)) else f"Class {c} DSC"
                ax0.plot(epochs[:train_arr.shape[0]], train_arr[:, c], label=label)
            ax0.set_xlabel("Epoch")
            ax0.set_ylabel("Training Dice Similarity Coefficient")
            ax0.legend()
        else:
            ax0.text(0.5, 0.5, "No per-class train dice logged", ha="center", va="center")
            ax0.set_axis_off()

        # right subplot: validation per-class dice
        ax1 = axs[1]
        ax1.set_title("Validation Dice Similarity Coefficients For Each Class")
        if val_arr is not None:
            num_classes = val_arr.shape[1]
            for c in range(num_classes):
                label = class_names[c] if (class_names and c < len(class_names)) else f"Class {c} DSC"
                ax1.plot(epochs[:val_arr.shape[0]], val_arr[:, c], label=label)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Validation Dice Similarity Coefficient")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No per-class val dice logged", ha="center", va="center")
            ax1.set_axis_off()

    else:
        # no log data at all
        for ax in axs:
            ax.text(0.5, 0.5, "No log data", ha="center", va="center")
            ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "dice_vs_epoch_by_class.png"), dpi=150)
    plt.close(fig)


# --------------------- per-case dice, for final plot ---------------------
def dice_per_class(pred_idx, gt_idx, num_classes, eps=1e-6):
    out = []
    for c in range(num_classes):
        p = (pred_idx == c)
        g = (gt_idx == c)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        d = (2.0 * inter + eps) / (denom + eps)
        out.append(float(d))
    return np.array(out, dtype=np.float32)


def save_triplet(vol, gt_idx, pr_idx, z, out_path):
    """
    Make 1x3 panel: Original | Label | Prediction at axial slice z.
    """
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


# --------------------- evaluation ---------------------
@torch.no_grad()
def evaluate_checkpoint(ckpt_path, save_dir, batch_size=1, max_triplets=12):
    os.makedirs(save_dir, exist_ok=True)

    # ---- load checkpoint / recover model args ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", NUM_CLASSES)

    target_dhw = tuple(ckpt_args.get("target", [96, 192, 192]))
    base_ch    = ckpt_args.get("base_ch", 32)
    depth      = ckpt_args.get("depth", 4)
    use_se     = not ckpt_args.get("no_se", False)
    use_att    = not ckpt_args.get("no_att", False)
    dropout    = ckpt_args.get("dropout", 0.0)
    train_ratio = ckpt_args.get("train_ratio", 0.8)
    seed        = ckpt_args.get("seed", 1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_ch=base_ch,
        depth=depth,
        use_se=use_se,
        use_att=use_att,
        dropout=dropout
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- build val loader ----
    all_pairs = find_pairs()
    _, val_pairs = split_by_patient(all_pairs, train_ratio=train_ratio, seed=seed)
    val_ds = Prostate3DDataset(val_pairs, target_dhw=target_dhw, one_hot=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---- load logs and generate training curves ----
    # class_names will be used in legends
    # adjust these to match your anatomical classes
    # order must match label indices 0..num_classes-1
    default_class_names = [
        "Background DSC",
        "Body DSC",
        "Bone DSC",
        "Bladder DSC",
        "Rectum DSC",
        "Prostate DSC"
    ]
    log_path = derive_log_path_from_ckpt(ckpt_path)
    log_data = load_training_log(log_path)
    plot_training_curves(log_data, save_dir, class_names=default_class_names)

    # ---- evaluate every validation case ----
    all_case_dice = []   # list of per-class dice arrays (len = num_classes)
    saved_triplets = 0
    case_idx_counter = 0

    for imgs, gts in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)  # (B,C,D,H,W)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()  # (B,D,H,W)
        gts_np = gts.cpu().numpy()

        for b in range(pred_idx.shape[0]):
            p = pred_idx[b]
            g = gts_np[b]
            vol = imgs[b, 0].cpu().numpy()

            per_class = dice_per_class(p, g, num_classes)  # shape (C,)
            all_case_dice.append(per_class)

            # save triplet for first N cases
            if saved_triplets < max_triplets:
                z = vol.shape[0] // 2
                out_path = os.path.join(
                    save_dir,
                    f"case_{case_idx_counter:03d}_triplet_z{z}.png"
                )
                save_triplet(vol, g, p, z, out_path)
                saved_triplets += 1

            case_idx_counter += 1

    all_case_dice = np.stack(all_case_dice, axis=0)  # (num_cases, C)

    # ---- summary numbers ----
    mean_per_class = all_case_dice.mean(axis=0)           # (C,)
    mean_dice = float(mean_per_class.mean())              # scalar
    n_cases = all_case_dice.shape[0]

    # ---- plot "Complete DICE Loss over testing"
    # we'll plot 1 - dice for each class across cases, plus total (1 - mean_dice_per_case)
    dice_loss_per_case = 1.0 - all_case_dice  # (num_cases, C)
    total_loss_per_case = 1.0 - all_case_dice.mean(axis=1)  # (num_cases,)

    plt.figure(figsize=(10,6))
    x_axis = np.arange(n_cases)
    # total first (like the blue curve in your screenshot)
    plt.plot(x_axis, total_loss_per_case, label="Total Loss")
    # now each class
    for c in range(num_classes):
        cname = f"Class {c} Loss"
        if c < len(default_class_names):
            # rename "Background DSC" -> "Class 0 Loss" style legend?
            # we'll keep "Class X Loss" to match screenshot, not "DSC"
            cname = f"{default_class_names[c].split()[0]} Loss" \
                    if default_class_names[c].endswith("DSC") else default_class_names[c]
        plt.plot(x_axis, dice_loss_per_case[:, c], label=cname)

    plt.title("Complete DICE Loss over testing")
    plt.xlabel("Case Index")
    plt.ylabel("DICE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_loss_over_cases.png"), dpi=150)
    plt.close()

    # ---- write short JSON summary
    summary = {
        "checkpoint": os.path.basename(ckpt_path),
        "n_cases": int(n_cases),
        "target_dhw": list(target_dhw),
        "mean_dice_overall": mean_dice,
        "mean_dice_per_class": mean_per_class.tolist(),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console print
    print("\n=== Evaluation Summary ===")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"#cases: {n_cases}")
    print(f"Mean Dice (overall): {mean_dice:.4f}")
    print("Per-class Dice:", np.round(mean_per_class, 4))
    print(f"Saved outputs to: {save_dir}")


# --------------------- entrypoint ---------------------
if __name__ == "__main__":
    ckpt_path = find_latest_ckpt("runs")
    run_tag = os.path.basename(ckpt_path).replace("_best.pt", "_eval")
    save_dir = os.path.join("results", run_tag)

    evaluate_checkpoint(
        ckpt_path=ckpt_path,
        save_dir=save_dir,
        batch_size=1,
        max_triplets=12
    )
