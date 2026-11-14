"""Dataset loader for processed concrete images."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ConcreteDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        self.project_root = Path(csv_path).resolve().parents[2]
        self.records = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        record = self.records[index]
        image_path = self.project_root / record["path"]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, record
