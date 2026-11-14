"""Generate small visual reference grids for Experiment 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import GaussianDiffusion
from model import SmallUNet


@torch.no_grad()
def sample(model, diffusion, count, class_id, device, image_size):
    images = torch.randn(count, 3, image_size, image_size, device=device)
    labels = torch.full((count,), class_id, device=device, dtype=torch.long)
    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((count,), timestep, device=device, dtype=torch.long)
        predicted_noise = model(images, t, labels)
        beta = diffusion.betas[timestep]
        alpha = diffusion.alphas[timestep]
        alpha_bar = diffusion.alpha_bars[timestep]
        mean = (images - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        images = mean if timestep == 0 else mean + torch.sqrt(beta) * torch.randn_like(images)
    return images.clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = SmallUNet(num_classes=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device)

    output_dir = root / "experiments" / "conditional_ddpm_128_augmented" / "outputs" / "reference_128"
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, class_id in (("cracked", 1), ("non_cracked", 0)):
        images = sample(model, diffusion, args.count, class_id, device, 128)
        output = output_dir / f"{class_name}_epoch_{checkpoint['epoch'] + 1:03d}.png"
        save_image(images, output, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"{class_name}: {output}")

    print(f"Device: {device}")
    print(f"Generated {args.count} images per class at 128x128")


if __name__ == "__main__":
    main()
