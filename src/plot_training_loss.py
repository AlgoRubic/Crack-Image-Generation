"""Plot unconditional and conditional DDPM training losses."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_loss(path: Path) -> tuple[list[int], list[float]]:
    epochs, losses = [], []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["mean_loss"]))
            except (KeyError, TypeError, ValueError):
                # Resumed runs can contain a repeated CSV header.
                continue
    return epochs, losses


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline_epochs, baseline_losses = read_loss(root / "outputs/logs/training_loss.csv")
    conditional_epochs, conditional_losses = read_loss(root / "outputs/logs/conditional_training_loss.csv")
    output_path = root / "outputs/plots/training_loss_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(baseline_epochs, baseline_losses, label="Unconditional DDPM", linewidth=2)
    plt.plot(conditional_epochs, conditional_losses, label="Conditional DDPM", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean noise-prediction MSE loss")
    plt.title("DDPM Training Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved loss plot: {output_path}")
    print(f"Unconditional final loss: {baseline_losses[-1]:.6f}")
    print(f"Conditional final loss: {conditional_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
