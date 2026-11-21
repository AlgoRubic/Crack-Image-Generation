"""
Fast 50-Step DDIM Sampler for Experiment 3 (128x128 Augmented Binary Conditional DDPM)
Generates 200 images per class (`non_cracked` and `cracked` = 400 total) inside:
  - `experiments/conditional_ddpm_128_augmented/outputs/evaluation/conditional_128_augmented_non_cracked`
  - `experiments/conditional_ddpm_128_augmented/outputs/evaluation/conditional_128_augmented_cracked`
Resumes seamlessly from existing images (`image_0000.png .. image_0063.png` in `cracked`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model import SmallUNet


@torch.no_grad()
def sample_ddim_128(
    model: SmallUNet,
    diffusion: GaussianDiffusion,
    count: int,
    class_id: int,
    device: torch.device,
    steps: int = 50,
    image_size: int = 128,
) -> torch.Tensor:
    images = torch.randn(count, 3, image_size, image_size, device=device)
    labels = torch.full((count,), class_id, device=device, dtype=torch.long)
    times = torch.linspace(diffusion.timesteps - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(len(times)):
        t = times[i]
        t_batch = torch.full((count,), t, device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            eps = model(images, t_batch, labels)

        alpha_bar_t = diffusion.alpha_bars[t]
        pred_x0 = (images - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if i < len(times) - 1:
            t_prev = times[i + 1]
            alpha_bar_prev = diffusion.alpha_bars[t_prev]
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps
            images = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            images = pred_x0

    return images.clamp(-1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/conditional_ddpm_128_augmented/checkpoints/conditional_128_augmented_epoch_050.pt",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallUNet(num_classes=2).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    exp_dir = ROOT / "experiments/conditional_ddpm_128_augmented"
    eval_base = exp_dir / "outputs/evaluation"

    for cid, cname in [(0, "non_cracked"), (1, "cracked")]:
        out_dir = eval_base / f"conditional_128_augmented_{cname}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        generated = 0
        batch_num = 0
        print(f"\n[*] Experiment 3: Sampling {args.count} images for Class {cid} ({cname}) at 128x128...")
        while generated < args.count:
            cur_n = min(args.batch_size, args.count - generated)
            paths = [out_dir / f"image_{generated + i:04d}.png" for i in range(cur_n)]
            if all(p.exists() for p in paths):
                print(f"  [{cname}] Batch {batch_num} ({generated}..{generated+cur_n-1}) exists, skipping...")
                generated += cur_n
                batch_num += 1
                continue

            seed = args.seed + cid * 1000 + batch_num
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            imgs = sample_ddim_128(model, diffusion, cur_n, cid, device, steps=args.steps, image_size=128).cpu()
            for img, p in zip(imgs, paths):
                save_image(img, p, normalize=True, value_range=(-1, 1))
            generated += cur_n
            batch_num += 1
            print(f"  [{cname}] Generated {generated}/{args.count} images ({time.time() - t0:.1f}s elapsed)")


if __name__ == "__main__":
    main()
