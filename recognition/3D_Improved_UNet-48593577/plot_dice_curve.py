import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_log(log_path):
    epochs = []
    train_dice = []
    val_dice = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # handle cases where keys might be missing
            e = row.get("epoch")
            td = row.get("train_dice")
            vd = row.get("val_dice")

            if e is None or td is None or vd is None:
                continue

            epochs.append(e)
            train_dice.append(td)
            val_dice.append(vd)

    return (
        np.array(epochs, dtype=np.float32),
        np.array(train_dice, dtype=np.float32),
        np.array(val_dice, dtype=np.float32),
    )

def plot_dice(epochs, train_dice, val_dice, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_dice, marker="o", label="Train Dice")
    plt.plot(epochs, val_dice, marker="o", label="Val Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.ylim(0, 1.05)
    plt.title("Training vs Validation Dice Coefficient per Epoch")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True,
                    help="Path to *_log.jsonl produced during training")
    ap.add_argument("--out", default=None,
                    help="Where to save the figure (default: alongside log)")
    args = ap.parse_args()

    # load arrays
    epochs, train_dice, val_dice = load_log(args.log)

    if len(epochs) == 0:
        print("No usable data found in log. Check that train_dice / val_dice exist.")
        return

    # default output location
    if args.out is None:
        base_dir = os.path.dirname(args.log)
        base_name = os.path.basename(args.log).replace("_log.jsonl", "")
        args.out = os.path.join(base_dir, f"{base_name}_train_val_dice_vs_epoch.png")

    plot_dice(epochs, train_dice, val_dice, args.out)

if __name__ == "__main__":
    main()
