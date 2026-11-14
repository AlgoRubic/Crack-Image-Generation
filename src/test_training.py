"""One-batch smoke test for the DDPM training path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ConcreteDataset
from diffusion import GaussianDiffusion
from model import SmallUNet


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ConcreteDataset(project_root / "data" / "processed" / "train.csv")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, records = next(iter(loader))
    images = images.to(device)
    timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)

    diffusion = GaussianDiffusion(device=device)
    model = SmallUNet().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    noisy_images, actual_noise = diffusion.add_noise(images, timesteps)
    predicted_noise = model(noisy_images, timesteps)
    loss = nn.functional.mse_loss(predicted_noise, actual_noise)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Device: {device}")
    print(f"Batch shape: {tuple(images.shape)}")
    print(f"Batch metadata fields: {list(records.keys())}")
    print(f"One-batch loss: {loss.item():.6f}")
    print("Training-path smoke test passed.")


if __name__ == "__main__":
    main()
