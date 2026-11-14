"""Train a small real-data crack classifier for generated-image evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class LabeledImageDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.root = csv_path.parents[2]
        with csv_path.open("r", encoding="utf-8") as handle:
            self.records = list(csv.DictReader(handle))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["label"] == "cracked")


class CrackClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    confusion = torch.zeros(2, 2, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            labels = labels.cpu()
            correct += (predictions == labels).sum().item()
            total += len(labels)
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1
    return correct / total, confusion


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_set = LabeledImageDataset(root / "data/processed/train.csv")
    validation_set = LabeledImageDataset(root / "data/processed/validation.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = torch.tensor([label for _, label in train_set])
    counts = torch.bincount(labels, minlength=2).float()
    weights = torch.tensor([1.0 / counts[label].item() for label in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=128, sampler=sampler, num_workers=2, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = CrackClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Classifier epoch {epoch + 1}"):
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        accuracy, confusion = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch + 1}: validation_accuracy={accuracy:.4f}")
        print(confusion)

    output = root / "checkpoints" / "crack_classifier.pt"
    torch.save({"model": model.state_dict(), "validation_accuracy": accuracy}, output)
    print(f"Saved classifier: {output}")


if __name__ == "__main__":
    main()
