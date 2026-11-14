"""Generate individual images for reproducible baseline evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import GaussianDiffusion
from model import SmallUNet
from sample import sample_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="unconditional_epoch_050")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "evaluation" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallUNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    generated = 0
    batch_number = 0
    while generated < args.count:
        current_batch = min(args.batch_size, args.count - generated)
        torch.manual_seed(args.seed + batch_number)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + batch_number)
        images = sample_images(model, diffusion, current_batch, device)
        for image in images:
            output_path = output_dir / f"image_{generated:04d}.png"
            save_image(image, output_path, normalize=True, value_range=(-1, 1))
            generated += 1
        batch_number += 1
        print(f"Generated {generated}/{args.count}")

    print(f"Device: {device}")
    print(f"Evaluation set: {output_dir}")


if __name__ == "__main__":
    main()
