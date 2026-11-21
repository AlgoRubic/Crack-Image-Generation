"""
Train Experiment 4: 13.8M AttentionUNet256 for Morphological Concrete Crack Generation.
- Native 256x256 resolution
- Batch size: 6 (~2.8 GB VRAM, ~1.8 GB System RAM)
- Balanced class sampling across all 5 classes
- FP16 AMP with AdamW
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model_attention import AttentionUNet256


class CrackTypeDataset256(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.records = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rec = self.records[index]
        img_path = ROOT / rec["path"]
        with Image.open(img_path) as im:
            tensor = self.transform(im.convert("RGB"))
        class_id = int(rec["class_id"])
        return tensor, class_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint or 'latest' to resume training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_csv = ROOT / "data/processed_256/train.csv"
    dataset = CrackTypeDataset256(data_csv)

    # Class-balanced sampling
    labels = [int(r["class_id"]) for r in dataset.records]
    class_counts = torch.bincount(torch.tensor(labels), minlength=5).float()
    print("Class sample counts in dataset:", [int(c.item()) for c in class_counts])
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = torch.tensor([class_weights[lbl].item() for lbl in labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = AttentionUNet256(base_channels=40, num_classes=5, cfg_dropout=0.10).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,} ({params/1e6:.2f} M)")

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = ROOT / "experiments" / "crack_type_conditional"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_loss.csv"

    start_epoch = 1
    if args.resume:
        resume_target = args.resume
        if resume_target == "latest":
            ckpts = sorted(ckpt_dir.glob("crack_type_256_epoch_*.pt"))
            if ckpts:
                resume_target = str(ckpts[-1])
            else:
                print("[-] No checkpoints found to resume from. Starting from scratch.")
                resume_target = ""

        if resume_target:
            resume_path = Path(resume_target)
            if not resume_path.is_absolute():
                resume_path = ROOT / resume_path
            if resume_path.exists():
                print(f"[*] Resuming from checkpoint: {resume_path.name}")
                ckpt = torch.load(resume_path, map_location=device)
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                print(f"[*] Resuming at Epoch {start_epoch:02d}/{args.epochs:02d}")
            else:
                print(f"[-] Checkpoint file not found: {resume_path}")

    csv_mode = "a" if (args.resume and log_file.exists()) else "w"
    write_header = (csv_mode == "w")
    with log_file.open(csv_mode, newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        if write_header:
            writer.writerow(["epoch", "mean_loss", "epoch_time_sec"])

        print(f"\n--- Training Phase (Epochs {start_epoch:02d} to {args.epochs:02d}) ---")
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            t0 = time.time()
            pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}")

            for images, class_ids in pbar:
                images = images.to(device, non_blocking=True)
                class_ids = class_ids.to(device, non_blocking=True)
                timesteps = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    noisy_images, noise = diffusion.add_noise(images, timesteps)
                    pred_noise = model(noisy_images, timesteps, class_ids)
                    loss = nn.functional.mse_loss(pred_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})

            mean_loss = total_loss / len(loader)
            dt = time.time() - t0
            writer.writerow([epoch, mean_loss, dt])
            lf.flush()

            # Save checkpoint each epoch
            ckpt_path = ckpt_dir / f"crack_type_256_epoch_{epoch:03d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": mean_loss,
            }, ckpt_path)
            print(f"Epoch {epoch:02d}: mean_loss={mean_loss:.6f} ({dt:.1f}s)")

    print(f"\n[SUCCESS] Completed all {args.epochs} epochs!")


if __name__ == "__main__":
    main()
