"""Smoke test for the DDPM U-Net."""

from __future__ import annotations

import torch

from model import SmallUNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    images = torch.randn(4, 3, 64, 64, device=device)
    timesteps = torch.randint(0, 1000, (4,), device=device)
    predicted_noise = model(images, timesteps)

    print(f"Device: {device}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(predicted_noise.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    assert predicted_noise.shape == images.shape
    assert torch.isfinite(predicted_noise).all()
    print("U-Net smoke test passed.")


if __name__ == "__main__":
    main()
