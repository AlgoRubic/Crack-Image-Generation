"""
Generate 16 non-cracked samples at 256x256 using batch_size=8 (2 batches of 8).
Stays at ~1.8 GB VRAM, avoiding any PCIe paging/thrashing.
Takes ~5-6 minutes total.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(r"D:\Downloads\concrete_ddpm")
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model import SmallUNet

OUT_DIR = Path(r"D:\ANTIG\concrete_ddpm_scratch\samples_256")
CKPT_PATH = ROOT / "experiments" / "conditional_ddpm_128_augmented" / "checkpoints" / "conditional_128_augmented_epoch_050.pt"

@torch.no_grad()
def sample_batch(model, diffusion, count, class_id, device, size=256):
    images = torch.randn(count, 3, size, size, device=device)
    labels = torch.full((count,), class_id, device=device, dtype=torch.long)
    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((count,), timestep, device=device, dtype=torch.long)
        pred_noise = model(images, t, labels)
        beta = diffusion.betas[timestep]
        alpha = diffusion.alphas[timestep]
        alpha_bar = diffusion.alpha_bars[timestep]
        mean = (images - beta / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        images = mean if timestep == 0 else mean + torch.sqrt(beta) * torch.randn_like(images)
    return images.clamp(-1, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = SmallUNet(num_classes=2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device)

    class_dir = OUT_DIR / "non_cracked"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    base_seed = 3026
    
    # 2 batches of 8 = 16 images
    for b in range(2):
        seed = base_seed + b
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"Generating non-cracked sub-batch {b+1}/2 (8 images, seed {seed})...", flush=True)
        t0 = time.time()
        imgs = sample_batch(model, diffusion, count=8, class_id=0, device=device, size=256)
        print(f"  Sub-batch {b+1} done in {time.time()-t0:.1f}s", flush=True)
        all_images.append(imgs)
        for i, img in enumerate(imgs):
            idx = b * 8 + i
            save_image(img, class_dir / f"image_{idx:04d}.png", normalize=True, value_range=(-1, 1))

    all_tensor = torch.cat(all_images, dim=0)
    grid_path = OUT_DIR / "non_cracked_256_grid.png"
    save_image(all_tensor, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved non-cracked grid to: {grid_path}", flush=True)
    print("Non-cracked 256x256 generation complete!", flush=True)

if __name__ == "__main__":
    main()
