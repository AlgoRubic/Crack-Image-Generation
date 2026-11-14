"""Smoke test for the DDPM forward diffusion schedule."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import GaussianDiffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "train"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "plots"


def main() -> None:
    image_path = next(IMAGE_ROOT.glob("*.png"))
    image = Image.open(image_path).convert("RGB")
    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    clean_image = to_tensor(image).unsqueeze(0)

    diffusion = GaussianDiffusion(timesteps=1000)
    selected_timesteps = [0, 100, 300, 600, 900, 999]

    print(f"Input image: {image_path}")
    print(f"Input tensor shape: {tuple(clean_image.shape)}")
    print("Noise-level checks:")

    for timestep in selected_timesteps:
        timestep_tensor = torch.tensor([timestep], dtype=torch.long)
        noisy_image, noise = diffusion.add_noise(clean_image, timestep_tensor)
        print(
            f"  t={timestep:3d} | noisy mean={noisy_image.mean().item(): .4f}"
            f" | noisy std={noisy_image.std().item(): .4f}"
            f" | noise std={noise.std().item(): .4f}"
        )

    print("Forward diffusion smoke test passed.")


if __name__ == "__main__":
    main()
