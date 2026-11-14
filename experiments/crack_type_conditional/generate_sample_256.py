"""
Generate 16 cracked and 16 non-cracked samples at 256x256 resolution
using the Experiment 3 epoch 50 checkpoint.
Outputs are stored strictly under D:\ANTIG\concrete_ddpm_scratch\samples_256
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
        if timestep % 200 == 0:
            print(f"  Step {timestep}/1000...", flush=True)
    return images.clamp(-1, 1)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SmallUNet(num_classes=2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device)

    for class_name, class_id, seed in [("cracked", 1, 2026), ("non_cracked", 0, 3026)]:
        class_dir = OUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating 16 samples for {class_name} at 256x256 (seed {seed}) ---", flush=True)
        
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            
        t0 = time.time()
        # Generate batch of 16
        images = sample_batch(model, diffusion, count=16, class_id=class_id, device=device, size=256)
        dt = time.time() - t0
        print(f"Generated 16 {class_name} images in {dt:.1f}s ({dt/60:.2f} min)", flush=True)
        
        # Save individual images
        for i, img in enumerate(images):
            save_image(img, class_dir / f"image_{i:04d}.png", normalize=True, value_range=(-1, 1))
            
        # Save 4x4 grid
        grid_path = OUT_DIR / f"{class_name}_256_grid.png"
        save_image(images, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved {class_name} grid to: {grid_path}", flush=True)

    print("\nAll 256x256 sample generation complete!", flush=True)

if __name__ == "__main__":
    main()
