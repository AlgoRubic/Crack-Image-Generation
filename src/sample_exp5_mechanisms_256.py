"""
Sample 256x256 images for all 7 Structural Failure Mechanism classes in Experiment 5 (Level B)
using fast 50-step DDIM with Classifier-Free Guidance (CFG).
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
from sample_crack_type_256 import sample_ddim_cfg_batch


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
    model = AttentionUNet256(base_channels=40, num_classes=7).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "experiments" / "structural_mechanism_ddpm" / "generated_samples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [
        (0, "non_cracked"),
        (1, "flexural"),
        (2, "flexure_shear"),
        (3, "web_shear"),
        (4, "torsion"),
        (5, "corrosion_splitting"),
        (6, "shrinkage_map"),
    ]

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for cid, cname in classes:
        print(f"Sampling {args.count} images for Class {cid} ({cname}) with CFG scale {args.cfg_scale}...")
        imgs = sample_ddim_cfg_batch(
            model,
            diffusion,
            args.count,
            cid,
            device,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sub_batch_size=4,
        )

        class_dir = out_dir / cname
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(imgs.shape[0]):
            save_image(imgs[idx], class_dir / f"sample_{idx:03d}.png", normalize=True, value_range=(-1, 1))

        grid_path = out_dir / f"{cname}_256_grid.png"
        save_image(imgs, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"  Saved grid: {grid_path}")


if __name__ == "__main__":
    main()
