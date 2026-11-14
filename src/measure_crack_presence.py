"""Measure generated crack-presence rate using the validated classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from train_crack_classifier import CrackClassifier


class GeneratedDataset(Dataset):
    def __init__(self, folder: Path) -> None:
        self.paths = sorted(folder.glob("*.png"))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrackClassifier().to(device)
    checkpoint = torch.load(root / "checkpoints/crack_classifier.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(GeneratedDataset(root / args.folder), batch_size=128, num_workers=2)
    counts = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        for images in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            counts += torch.bincount(predictions, minlength=2)
    total = counts.sum().item()
    print(f"Images: {total}")
    print(f"Predicted non-cracked: {counts[0].item()}")
    print(f"Predicted cracked: {counts[1].item()}")
    print(f"Crack-presence rate: {counts[1].item() / total:.4f}")


if __name__ == "__main__":
    main()
