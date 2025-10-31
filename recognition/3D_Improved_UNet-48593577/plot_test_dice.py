import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_test_per_class_dice(summary_path, label_names=None):
    # Load summary
    with open(summary_path, "r") as f:
        summary = json.load(f)

    dice = np.array(summary["mean_dice_per_class"], dtype=np.float32)
    mean_dice = float(summary["mean_dice"])
    num_classes = len(dice)

    # Label names (optional)
    if label_names is None or len(label_names) != num_classes:
        label_names = [f"Class {i}" for i in range(num_classes)]

    # Plot
    plt.figure(figsize=(7, 5))
    bars = plt.bar(np.arange(num_classes), dice, color="skyblue", edgecolor="black")

    # Highlight mean
    plt.axhline(mean_dice, color="red", linestyle="--", label=f"Mean Dice = {mean_dice:.3f}")

    plt.xticks(np.arange(num_classes), label_names, rotation=20, ha="right")
    plt.ylabel("Dice Coefficient")
    plt.ylim(0, 1.05)
    plt.title("Per-Class Dice Scores on Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, axis="y")

    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{dice[i]:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(summary_path), "test_per_class_dice.png")
    plt.savefig(out_path, dpi=200)
    print(f"✅ Saved: {out_path}")

    return dice, mean_dice


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True,
                    help="Path to summary.json output by predict.py")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional label names for each class (space-separated)")
    args = ap.parse_args()

    plot_test_per_class_dice(args.summary, label_names=args.labels)
