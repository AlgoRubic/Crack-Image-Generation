"""Generate images from a trained DDPM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import GaussianDiffusion
from model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def sample_images(
    model: SmallUNet,
    diffusion: GaussianDiffusion,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    images = torch.randn(count, 3, 64, 64, device=device)

    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((count,), timestep, device=device, dtype=torch.long)
        predicted_noise = model(images, t)

        beta = diffusion.betas[timestep]
        alpha = diffusion.alphas[timestep]
        alpha_bar = diffusion.alpha_bars[timestep]

        # DDPM posterior mean for x_{t-1} given x_t and predicted noise.
        mean = (1.0 / torch.sqrt(alpha)) * (
            images - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        )

        if timestep > 0:
            noise = torch.randn_like(images)
            images = mean + torch.sqrt(beta) * noise
        else:
            images = mean

    return images.clamp(-1.0, 1.0)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = SmallUNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    images = sample_images(model, diffusion, args.count, device)
    output_path = output_dir / f"samples_epoch_{checkpoint['epoch'] + 1:03d}_seed_{args.seed}.png"
    save_image(images, output_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    print(f"Generated images: {args.count}")
    print(f"Saved grid: {output_path}")


if __name__ == "__main__":
    main()
