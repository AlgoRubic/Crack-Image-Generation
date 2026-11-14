"""Train the small unconditional DDPM on processed concrete images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ConcreteDataset
from diffusion import GaussianDiffusion
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def save_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_csv = project_root / "data" / "processed" / "train.csv"
    checkpoint_root = project_root / "checkpoints"
    log_path = project_root / "outputs" / "logs" / "training_loss.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ConcreteDataset(train_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = SmallUNet().to(device)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed checkpoint: {args.resume}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or start_epoch == 0
    with log_path.open("a", newline="", encoding="utf-8") as log_handle:
        log_writer = csv.writer(log_handle)
        if write_header:
            log_writer.writerow(["epoch", "mean_loss"])

        print(f"Device: {device}")
        print(f"Images: {len(dataset)}")

        for epoch in range(start_epoch, start_epoch + args.epochs):
            model.train()
            total_loss = 0.0
            batches_seen = 0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}")

            for batch_index, (clean_images, _) in enumerate(progress):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break

                clean_images = clean_images.to(device, non_blocking=True)
                timesteps = torch.randint(
                    0, diffusion.timesteps, (clean_images.shape[0],), device=device
                )
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    noisy_images, actual_noise = diffusion.add_noise(clean_images, timesteps)
                    predicted_noise = model(noisy_images, timesteps)
                    loss = nn.functional.mse_loss(predicted_noise, actual_noise)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                batches_seen += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")

            if batches_seen == 0:
                raise RuntimeError("No training batches were processed.")

            mean_loss = total_loss / batches_seen
            log_writer.writerow([epoch + 1, mean_loss])
            log_handle.flush()
            checkpoint_path = checkpoint_root / f"ddpm_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, epoch, mean_loss)
            print(f"Epoch {epoch + 1}: mean_loss={mean_loss:.6f}")
            print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
