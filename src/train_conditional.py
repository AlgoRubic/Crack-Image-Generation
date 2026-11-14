"""Train a crack-status conditional DDPM from scratch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ConcreteDataset
from diffusion import GaussianDiffusion
from model import SmallUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = ConcreteDataset(root / "data" / "processed" / "train.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Balance the rare cracked class during conditional training.
    labels = [1 if record["label"] == "cracked" else 0 for record in dataset.records]
    class_counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    sample_weights = torch.tensor([1.0 / class_counts[label].item() for label in labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = SmallUNet(num_classes=2).to(device)
    diffusion = GaussianDiffusion(device=device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = root / "checkpoints" / "conditional"
    log_path = root / "outputs" / "logs" / "conditional_training_loss.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        writer = csv.writer(log_handle)
        writer.writerow(["epoch", "mean_loss"])
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for clean_images, records in tqdm(loader, desc=f"Conditional epoch {epoch + 1}"):
                clean_images = clean_images.to(device, non_blocking=True)
                class_labels = torch.tensor(
                    [1 if label == "cracked" else 0 for label in records["label"]],
                    device=device,
                    dtype=torch.long,
                )
                timesteps = torch.randint(0, diffusion.timesteps, (clean_images.shape[0],), device=device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    noisy_images, actual_noise = diffusion.add_noise(clean_images, timesteps)
                    predicted_noise = model(noisy_images, timesteps, class_labels)
                    loss = nn.functional.mse_loss(predicted_noise, actual_noise)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            mean_loss = total_loss / len(loader)
            writer.writerow([epoch + 1, mean_loss])
            log_handle.flush()
            checkpoint_path = checkpoint_dir / f"conditional_epoch_{epoch + 1:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": mean_loss}, checkpoint_path)
            print(f"Epoch {epoch + 1}: mean_loss={mean_loss:.6f}")


if __name__ == "__main__":
    main()
