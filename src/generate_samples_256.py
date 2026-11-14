"""
Generate 16 cracked and 16 non-cracked 256x256 samples using batch_size=8.
Safe memory footprint (~1.8 GB VRAM), avoiding PCIe/RAM paging.
Saves individual images as soon as each batch of 8 completes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model import SmallUNet

OUT_DIR = ROOT / "experiments" / "crack_type_conditional" / "samples_256"
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
            print(f"    Step {timestep:4d}/1000", flush=True)
    return images.clamp(-1, 1)

def generate_condition(model, diffusion, class_name, class_id, base_seed, device, batch_size=8, total_count=16):
    class_dir = OUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==========================================", flush=True)
    print(f"Generating {total_count} {class_name} samples at 256x256", flush=True)
    print(f"Batch size: {batch_size}, Base seed: {base_seed}", flush=True)
    print(f"==========================================", flush=True)
    
    all_images = []
    num_batches = (total_count + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        count = min(batch_size, total_count - len(all_images) * batch_size)
        seed = base_seed + b
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            
        print(f"\n  Batch {b + 1}/{num_batches}: generating {count} images (seed {seed})...", flush=True)
        t0 = time.time()
        imgs = sample_batch(model, diffusion, count, class_id, device, size=256)
        dt = time.time() - t0
        print(f"  Batch {b + 1}/{num_batches} done in {dt:.1f}s ({dt/60:.2f} min)", flush=True)
        
        # Save individual images immediately
        start_idx = b * batch_size
        for i, img in enumerate(imgs):
            img_path = class_dir / f"image_{start_idx + i:04d}.png"
            save_image(img, img_path, normalize=True, value_range=(-1, 1))
            print(f"    Saved: {img_path.name}", flush=True)
        all_images.append(imgs)

    combined = torch.cat(all_images, dim=0)
    grid_path = OUT_DIR / f"{class_name}_256_grid.png"
    save_image(combined, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved {class_name} 4x4 grid to: {grid_path}", flush=True)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"Loading checkpoint: {CKPT_PATH.name}...", flush=True)
    model = SmallUNet(num_classes=2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device)

    # 1. Cracked: 16 images (2 batches of 8, seeds 2026, 2027)
    generate_condition(model, diffusion, "cracked", 1, base_seed=2026, device=device)
    
    # 2. Non-cracked: 16 images (2 batches of 8, seeds 3026, 3027)
    generate_condition(model, diffusion, "non_cracked", 0, base_seed=3026, device=device)

    print("\n[SUCCESS] Completed all 16 cracked and 16 non-cracked 256x256 samples!", flush=True)
    print(f"Outputs saved to: {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
