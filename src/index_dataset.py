"""Index SDNET2018 images and create reproducible train/validation CSV files."""

from __future__ import annotations

import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "SDNET2018" / "images"
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
SEED = 42
VALIDATION_RATIO = 0.10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def decode_folder_labels(path: Path) -> tuple[str, str]:
    """Decode SDNET2018 folder names such as D/CD and W/UW."""
    surface_code = path.parent.name
    class_code = path.name

    surface_names = {
        "D": "bridge_deck",
        "P": "pavement",
        "W": "wall",
    }

    if surface_code not in surface_names:
        raise ValueError(f"Unknown surface folder: {surface_code}")

    if not class_code or class_code[0] not in {"C", "U"}:
        raise ValueError(f"Unknown class folder: {path}")

    crack_label = "cracked" if class_code[0] == "C" else "non_cracked"
    return surface_names[surface_code], crack_label


def collect_records() -> list[dict[str, str]]:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Dataset directory not found: {IMAGE_ROOT}")

    records: list[dict[str, str]] = []
    for image_path in sorted(IMAGE_ROOT.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        surface, crack_label = decode_folder_labels(image_path.parent)
        records.append(
            {
                "path": relative_path,
                "surface": surface,
                "label": crack_label,
            }
        )

    if not records:
        raise RuntimeError(f"No image files found below {IMAGE_ROOT}")

    return records


def split_records(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Stratify by surface and crack label for balanced train/validation splits."""
    rng = random.Random(SEED)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}

    for record in records:
        key = (record["surface"], record["label"])
        groups.setdefault(key, []).append(record)

    train_records: list[dict[str, str]] = []
    validation_records: list[dict[str, str]] = []

    for group in groups.values():
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * VALIDATION_RATIO))
        validation_records.extend(group[:validation_count])
        train_records.extend(group[validation_count:])

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    return train_records, validation_records


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "surface", "label"])
        writer.writeheader()
        writer.writerows(records)


def print_summary(name: str, records: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["surface"], record["label"])
        counts[key] = counts.get(key, 0) + 1

    print(f"{name}: {len(records)} images")
    for (surface, label), count in sorted(counts.items()):
        print(f"  {surface:14} {label:12} {count}")


def main() -> None:
    records = collect_records()
    train_records, validation_records = split_records(records)

    write_csv(SPLIT_ROOT / "all.csv", records)
    write_csv(SPLIT_ROOT / "train.csv", train_records)
    write_csv(SPLIT_ROOT / "validation.csv", validation_records)

    print_summary("All", records)
    print_summary("Train", train_records)
    print_summary("Validation", validation_records)
    print(f"CSV files written to: {SPLIT_ROOT}")


if __name__ == "__main__":
    main()
