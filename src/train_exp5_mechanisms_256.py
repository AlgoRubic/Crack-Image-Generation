"""
Train Experiment 5 (Level B): 7-Class Structural Failure Mechanism DDPM at Native 256x256.
Uses Transfer Learning from Experiment 4 (crack_type_256_epoch_050.pt) with semantic class-embedding
projection and structure-preserving augmentations (no vertical flips, preserving bottom-tension orientation).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model_attention import AttentionUNet256


class StructuralMechanismDataset256(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.records = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

        # Structure-preserving transform:
        # RandomHorizontalFlip preserves bottom-edge tension initiation and vertical/horizontal/diagonal physics.
        # RandomVerticalFlip is intentionally omitted so bottom-tension flexural cracks are never inverted!
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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


def load_transfer_weights(model: AttentionUNet256, exp4_ckpt_path: Path, device: torch.device) -> None:
    """
    Load 100% of the U-Net backbone weights from Experiment 4 (5 classes + 1 null = 6 tokens)
    and semantically map the learned class embeddings into Experiment 5 (7 classes + 1 null = 8 tokens).
    """
    print(f"[*] Loading Experiment 4 pretrained backbone from: {exp4_ckpt_path.name}")
    ckpt = torch.load(exp4_ckpt_path, map_location=device)
    old_state = ckpt["model"]
    new_state = model.state_dict()

    for k, v in old_state.items():
        if k == "class_embed.weight":
            # old_state["class_embed.weight"] has shape (6, 160):
            #   0: non_cracked, 1: longitudinal, 2: transverse, 3: diagonal, 4: alligator_map, 5: null
            # new_state["class_embed.weight"] has shape (8, 160):
            #   0: non_cracked, 1: flexural, 2: flexure_shear, 3: web_shear,
            #   4: torsion, 5: corrosion_splitting, 6: shrinkage_map, 7: null
            with torch.no_grad():
                emb_new = torch.zeros_like(new_state[k])
                emb_new[0] = v[0]  # non_cracked <- non_cracked
                emb_new[1] = v[2]  # flexural (vertical bottom-tension) <- transverse
                emb_new[2] = 0.5 * (v[2] + v[3])  # flexure_shear (vertical bending to diagonal)
                emb_new[3] = v[3]  # web_shear (diagonal tension) <- diagonal
                emb_new[4] = 0.7 * v[3] + 0.3 * v[4]  # torsion (parallel echelon diagonal network)
                emb_new[5] = v[1]  # corrosion_splitting (horizontal rebar axis) <- longitudinal
                emb_new[6] = v[4]  # shrinkage_map <- alligator_map
                emb_new[7] = v[5]  # CFG null token <- CFG null token
                new_state[k] = emb_new
        elif k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v

    model.load_state_dict(new_state)
    print("[*] Successfully transferred 13.82M weights + semantic mechanism embeddings!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pretrained",
        type=str,
        default="experiments/crack_type_conditional/checkpoints/crack_type_256_epoch_050.pt",
        help="Experiment 4 checkpoint for transfer learning initialization",
    )
    parser.add_argument("--resume", type=str, default="", help="Path to Exp 5 checkpoint or 'latest'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    csv_path = ROOT / "data" / "processed_exp5_mechanisms" / "train.csv"
    dataset = StructuralMechanismDataset256(csv_path)

    num_classes = 7
    class_counts = [0] * num_classes
    for r in dataset.records:
        class_counts[int(r["class_id"])] += 1
    print(f"Class sample counts (0..6): {class_counts}")

    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[int(r["class_id"])] for r in dataset.records]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = AttentionUNet256(
        in_channels=3,
        out_channels=3,
        base_channels=40,
        num_classes=num_classes,
        cfg_dropout=0.15,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,} ({params / 1e6:.2f} M)")

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = ROOT / "experiments" / "structural_mechanism_ddpm"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_loss.csv"

    start_epoch = 1
    resumed = False
    if args.resume:
        resume_target = args.resume
        if resume_target == "latest":
            ckpts = sorted(ckpt_dir.glob("exp5_mechanism_256_epoch_*.pt"))
            resume_target = str(ckpts[-1]) if ckpts else ""
        if resume_target:
            resume_path = Path(resume_target)
            if not resume_path.is_absolute():
                resume_path = ROOT / resume_path
            if resume_path.exists():
                print(f"[*] Resuming Experiment 5 from checkpoint: {resume_path.name}")
                ckpt = torch.load(resume_path, map_location=device)
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                resumed = True
                print(f"[*] Resuming at Epoch {start_epoch:02d}/{args.epochs:02d}")

    if not resumed and args.pretrained:
        pre_path = ROOT / args.pretrained
        if pre_path.exists():
            load_transfer_weights(model, pre_path, device)

    csv_mode = "a" if (resumed and log_file.exists()) else "w"
    write_header = (csv_mode == "w")
    with log_file.open(csv_mode, newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        if write_header:
            writer.writerow(["epoch", "mean_loss", "epoch_time_sec"])

        print(f"\n--- Experiment 5 Training Phase (Epochs {start_epoch:02d} to {args.epochs:02d}) ---")
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

            ckpt_path = ckpt_dir / f"exp5_mechanism_256_epoch_{epoch:03d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": mean_loss,
            }, ckpt_path)
            print(f"Epoch {epoch:02d}: mean_loss={mean_loss:.6f} ({dt:.1f}s)")

    print(f"\n[SUCCESS] Completed all {args.epochs} epochs of Experiment 5!")


if __name__ == "__main__":
    main()
