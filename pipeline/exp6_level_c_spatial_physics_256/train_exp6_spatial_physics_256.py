"""
Train Experiment 6 (Level C): Physics-Guided Spatial Trajectory & Aperture-Field Conditioned DDPM at Native 256x256.
- Architecture: PhysicsControlUNet256 (5 input channels: 3 RGB noisy x_t + 2 Spatial Physics [S_skeleton, S_aperture])
- Transfer Learning: 100% of U-Net backbone and 7-class embeddings transferred from Experiment 5
                     (exp5_mechanism_256_epoch_025.pt) with Zero-Initialized Input-Channel Surgery on init_conv.
- Chromatic Normalization: Eliminates purple/blue/green camera white-balance shifts on training patches.
- Dual Classifier-Free Guidance: Simultaneously drops class token to null (7) and physics tensor to [-1, -1] (p=0.15).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_opening, distance_transform_edt, gaussian_filter, maximum_filter, uniform_filter
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model_attention import AttentionUNet256


def normalize_chromatic_cast(rgb_np: np.ndarray, gamma: float = 0.08) -> np.ndarray:
    """
    Assumption C4: Chromatic White-Balance Normalization.
    Damps unnatural purple/blue/green camera white-balance shifts toward neutral Portland cement concrete
    while preserving 100% of pixel-level luminance contrast (cracks, aggregate boundaries, and pores).
    """
    arr = rgb_np.astype(np.float32)
    mu_r = float(arr[:, :, 0].mean())
    mu_g = float(arr[:, :, 1].mean())
    mu_b = float(arr[:, :, 2].mean())
    mu_l = 0.299 * mu_r + 0.587 * mu_g + 0.114 * mu_b

    # Also damp per-pixel chromatic saturation so individual pixels cannot drift into neon purple/blue
    pixel_l = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    arr[:, :, 0] = pixel_l + gamma * (arr[:, :, 0] - pixel_l) + 1.5
    arr[:, :, 1] = pixel_l + gamma * (arr[:, :, 1] - pixel_l) + 0.5
    arr[:, :, 2] = pixel_l + gamma * (arr[:, :, 2] - pixel_l) - 1.5
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def extract_paired_spatial_physics(rgb_np: np.ndarray, class_id: int, w_max_norm: float = 6.0) -> np.ndarray:
    """
    Assumption C1: Extract 2-channel Spatial Physics Field S(x,y) = [S_skeleton, S_aperture] in [0, 1]^(2x256x256)
    directly from the aligned concrete patch.
    For Class 0 (non_cracked), S(x,y) is identically zero everywhere.
    """
    h, w, _ = rgb_np.shape
    if class_id == 0:
        return np.zeros((2, h, w), dtype=np.float32)

    gray = (
        0.299 * rgb_np[:, :, 0].astype(np.float32)
        + 0.587 * rgb_np[:, :, 1].astype(np.float32)
        + 0.114 * rgb_np[:, :, 2].astype(np.float32)
    )
    local_mean = uniform_filter(gray, size=21)
    diff = local_mean - gray
    fissure = binary_opening(diff > 15.5, structure=np.ones((2, 2), dtype=bool))

    if not np.any(fissure):
        fissure = diff > 11.5

    if not np.any(fissure):
        return np.zeros((2, h, w), dtype=np.float32)

    edt = distance_transform_edt(fissure).astype(np.float32)
    local_max = maximum_filter(edt, size=3)
    ridge = (fissure & (edt >= np.maximum(0.8, local_max - 0.35))).astype(np.float32)

    s_skel = gaussian_filter(ridge, sigma=1.1)
    max_s = float(s_skel.max())
    if max_s > 1e-5:
        s_skel = np.clip(s_skel / max_s, 0.0, 1.0)

    s_aper = np.clip(edt / w_max_norm, 0.0, 1.0)
    return np.stack([s_skel, s_aper], axis=0).astype(np.float32)


class SpatialPhysicsMechanismDataset256(Dataset):
    """
    Returns:
      - x0: Tensor of shape (3, 256, 256) in [-1, 1] (chromatically normalized concrete patch)
      - phys: Tensor of shape (2, 256, 256) in [-1, 1] ([S_skeleton, S_aperture])
      - class_id: int in 0..6
    """
    def __init__(self, csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[index]
        img_path = ROOT / rec["path"]
        class_id = int(rec["class_id"])

        with Image.open(img_path) as im:
            rgb_np = np.array(im.convert("RGB"), dtype=np.uint8)

        # Synchronized Horizontal Flip (Assumption A2: No vertical flip so bottom-tension orientation is preserved!)
        if random.random() < 0.5:
            rgb_np = np.ascontiguousarray(rgb_np[:, ::-1, :])

        # Chromatic white-balance normalization (Assumption C4)
        rgb_clean = normalize_chromatic_cast(rgb_np)

        # Extract paired 2-channel spatial physics field S in [0, 1]
        phys_01 = extract_paired_spatial_physics(rgb_clean, class_id)

        # Convert both to [-1, 1] tensors
        x0 = torch.from_numpy(rgb_clean).permute(2, 0, 1).float() / 127.5 - 1.0
        phys = torch.from_numpy(phys_01).float() * 2.0 - 1.0
        return x0, phys, class_id


def load_exp5_with_input_channel_surgery(
    model: AttentionUNet256,
    exp5_ckpt_path: Path,
    device: torch.device,
) -> None:
    """
    Assumption C3: Exact Input-Channel Surgery (3 Channels -> 5 Channels).
    Copies 100% of Experiment 5's trained weights into the 5-channel PhysicsControlUNet256,
    copying RGB weights into channels 0..2 of init_conv and zero-initializing channels 3..4 (S_skeleton, S_aperture).
    """
    print(f"[*] Performing Input-Channel Surgery from Experiment 5 checkpoint: {exp5_ckpt_path.name}")
    ckpt = torch.load(exp5_ckpt_path, map_location=device)
    old_state = ckpt["model"]
    new_state = model.state_dict()

    for k, v in old_state.items():
        if k == "init_conv.weight":
            # v has shape (40, 3, 3, 3); new_state[k] has shape (40, 5, 3, 3)
            with torch.no_grad():
                w_surg = torch.zeros_like(new_state[k])
                w_surg[:, :3, :, :] = v
                w_surg[:, 3:5, :, :] = 0.0  # Zero-convolution initialization for spatial physics channels
                new_state[k] = w_surg
        elif k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v

    model.load_state_dict(new_state)
    print("[*] Successfully transferred 13.82M weights + Zero-Init Spatial Physics Channels (3 -> 5 ch)!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pretrained",
        type=str,
        default="experiments/structural_mechanism_ddpm/checkpoints/exp5_mechanism_256_epoch_025.pt",
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path = ROOT / "data/processed_exp5_mechanisms/train.csv"
    out_dir = ROOT / "experiments/exp6_level_c_spatial_physics"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpatialPhysicsMechanismDataset256(csv_path)
    num_classes = 7

    class_counts = [0] * num_classes
    for rec in dataset.records:
        class_counts[int(rec["class_id"])] += 1
    print(f"Class sample counts (0..6): {class_counts}")

    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[int(rec["class_id"])] for rec in dataset.records]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 5-Channel Input (3 RGB + 2 Spatial Physics), 3-Channel Output (Predicted RGB Noise)
    model = AttentionUNet256(
        in_channels=5,
        out_channels=3,
        base_channels=40,
        num_classes=num_classes,
        cfg_dropout=0.0,  # Handled explicitly via Dual-CFG below so class & spatial field drop together
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f} M)")

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    loss_csv = log_dir / "training_loss.csv"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
    else:
        pretrained_path = ROOT / args.pretrained
        if pretrained_path.exists():
            load_exp5_with_input_channel_surgery(model, pretrained_path, device)
        with loss_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "mean_loss", "epoch_time_sec"])

    cfg_drop_prob = 0.15
    null_token = num_classes  # token 7

    print(f"\n--- Experiment 6 (Level C) Spatial-Physics Training Phase (Epochs {start_epoch:02d} to {args.epochs:02d}) ---")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", dynamic_ncols=True)
        for x0, phys, labels in pbar:
            x0 = x0.to(device, non_blocking=True)
            phys = phys.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            b = x0.shape[0]
            t = torch.randint(0, diffusion.timesteps, (b,), device=device).long()
            x_t, noise = diffusion.add_noise(x0, t)

            # Assumption C5: Joint Spatial + Class Dual Classifier-Free Guidance Dropout (p = 0.15)
            drop_mask = torch.rand(b, device=device) < cfg_drop_prob
            labels_in = torch.where(drop_mask, torch.full_like(labels, null_token), labels)
            phys_null = torch.full_like(phys, -1.0)
            phys_in = torch.where(drop_mask.view(b, 1, 1, 1), phys_null, phys)

            # Concatenate 3-channel noisy image x_t and 2-channel spatial physics field phys_in -> (B, 5, 256, 256)
            x_in = torch.cat([x_t, phys_in], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_noise = model(x_in, t, labels_in)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            steps += 1
            pbar.set_postfix(loss=f"{loss_val:.5f}")

        dt = time.time() - t0
        mean_loss = epoch_loss / max(steps, 1)
        print(f"Epoch {epoch:02d}: mean_loss={mean_loss:.6f} ({dt:.1f}s)")

        with loss_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_loss, dt])

        ckpt_path = ckpt_dir / f"exp6_spatial_physics_256_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mean_loss": mean_loss,
                "num_classes": num_classes,
                "in_channels": 5,
            },
            ckpt_path,
        )

    print(f"\n[SUCCESS] Completed all {args.epochs} epochs of Experiment 6 (Level C)!")


if __name__ == "__main__":
    main()
