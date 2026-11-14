"""Calculate FID, KID and LPIPS for real and generated image folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class ImageFolderDataset(Dataset):
    def __init__(self, folder: Path) -> None:
        self.paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64), antialias=True)]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class PairedDataset(Dataset):
    def __init__(self, real: ImageFolderDataset, fake: ImageFolderDataset, count: int) -> None:
        self.real = torch.utils.data.Subset(real, range(count))
        self.fake = torch.utils.data.Subset(fake, range(count))

    def __len__(self) -> int:
        return len(self.real)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.real[index], self.fake[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="data/processed/validation")
    parser.add_argument("--fake", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    real_dataset = ImageFolderDataset(project_root / args.real)
    fake_dataset = ImageFolderDataset(project_root / args.fake)
    count = min(args.max_images, len(real_dataset), len(fake_dataset))
    real_loader = DataLoader(torch.utils.data.Subset(real_dataset, range(count)), batch_size=args.batch_size)
    fake_loader = DataLoader(torch.utils.data.Subset(fake_dataset, range(count)), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=64, subsets=10, subset_size=min(100, count), normalize=False).to(device)

    with torch.no_grad():
        for real_batch in real_loader:
            real_uint8 = (real_batch * 255).round().to(torch.uint8).to(device)
            fid.update(real_uint8, real=True)
            kid.update(real_uint8, real=True)
        for fake_batch in fake_loader:
            fake_uint8 = (fake_batch * 255).round().to(torch.uint8).to(device)
            fid.update(fake_uint8, real=False)
            kid.update(fake_uint8, real=False)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    pair_loader = DataLoader(PairedDataset(real_dataset, fake_dataset, count), batch_size=args.batch_size)
    lpips_values = []
    with torch.no_grad():
        for real_batch, fake_batch in pair_loader:
            lpips_values.append(lpips(real_batch.to(device) * 2 - 1, fake_batch.to(device) * 2 - 1))

    print(f"Device: {device}")
    print(f"Images evaluated: {count}")
    print(f"FID: {fid.compute().item():.6f}")
    kid_mean, kid_std = kid.compute()
    print(f"KID mean: {kid_mean.item():.6f}")
    print(f"KID std: {kid_std.item():.6f}")
    print(f"Paired LPIPS mean: {torch.stack(lpips_values).mean().item():.6f}")
    print("Note: paired LPIPS is a perceptual-distance diagnostic, not a standalone quality score.")


if __name__ == "__main__":
    main()
