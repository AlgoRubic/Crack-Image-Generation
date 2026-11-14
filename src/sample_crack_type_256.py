"""
Sample 256x256 images from trained AttentionUNet256 using fast 50-step DDIM with Classifier-Free Guidance (CFG).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model_attention import AttentionUNet256


@torch.no_grad()
def sample_ddim_cfg_batch(
    model: AttentionUNet256,
    diffusion: GaussianDiffusion,
    count: int,
    class_id: int,
    device: torch.device,
    size: int = 256,
    steps: int = 50,
    cfg_scale: float = 2.5,
    sub_batch_size: int = 4,
) -> torch.Tensor:
    all_imgs = []
    generated = 0
    while generated < count:
        cur_n = min(sub_batch_size, count - generated)
        images = torch.randn(cur_n, 3, size, size, device=device)
        labels = torch.full((cur_n,), class_id, device=device, dtype=torch.long)
        times = torch.linspace(0, diffusion.timesteps - 1, steps, dtype=torch.long, device=device).flip(0)

        for i in range(len(times)):
            t_cur = times[i]
            t_prev = times[i + 1] if i + 1 < len(times) else torch.tensor(-1, device=device)
            t = torch.full((cur_n,), t_cur, device=device, dtype=torch.long)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                eps_cond = model(images, t, labels, force_dropout=False)
                if cfg_scale > 1.0:
                    eps_uncond = model(images, t, labels, force_dropout=True)
                    pred_noise = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                else:
                    pred_noise = eps_cond

            alpha_bar_cur = diffusion.alpha_bars[t_cur]
            alpha_bar_prev = diffusion.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            pred_x0 = (images - torch.sqrt(1 - alpha_bar_cur) * pred_noise) / torch.sqrt(alpha_bar_cur)
            pred_x0 = pred_x0.clamp(-1, 1)

            if t_prev >= 0:
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
                images = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            else:
                images = pred_x0

        all_imgs.append(images.clamp(-1, 1).cpu())
        generated += cur_n

    return torch.cat(all_imgs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet256(base_channels=40, num_classes=5).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "experiments/crack_type_conditional/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [
        (0, "non_cracked"),
        (1, "longitudinal"),
        (2, "transverse"),
        (3, "diagonal"),
        (4, "alligator_map"),
    ]

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for cid, cname in classes:
        print(f"Sampling {args.count} images for Class {cid} ({cname}) with CFG scale {args.cfg_scale}...")
        imgs = sample_ddim_cfg_batch(model, diffusion, args.count, cid, device, steps=args.steps, cfg_scale=args.cfg_scale, sub_batch_size=4)
        
        class_dir = out_dir / cname
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(imgs.shape[0]):
            save_image(imgs[idx], class_dir / f"sample_{idx:03d}.png", normalize=True, value_range=(-1, 1))

        grid_path = out_dir / f"{cname}_256_grid.png"
        save_image(imgs, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"  Saved grid: {grid_path}")


if __name__ == "__main__":
    main()
