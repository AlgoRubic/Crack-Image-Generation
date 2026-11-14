"""Resize SDNET2018 images and prepare them for DDPM training."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
IMAGE_SIZE = (64, 64)


def stable_name(relative_path: str) -> str:
    """Create a collision-resistant filename from the original relative path."""
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.png"


def process_split(split_name: str) -> None:
    input_csv = SPLIT_ROOT / f"{split_name}.csv"
    output_image_root = OUTPUT_ROOT / split_name
    output_csv = OUTPUT_ROOT / f"{split_name}.csv"
    output_image_root.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    processed_records: list[dict[str, str]] = []
    skipped = 0

    for record in tqdm(records, desc=f"Processing {split_name}"):
        source_path = PROJECT_ROOT / Path(record["path"])
        output_name = stable_name(record["path"])
        output_path = output_image_root / output_name

        try:
            with Image.open(source_path) as image:
                # RGB guarantees three consistent channels for the U-Net input.
                image = image.convert("RGB")
                # LANCZOS preserves fine crack details better than nearest-neighbour resizing.
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                image.save(output_path, format="PNG", optimize=True)
        except (OSError, ValueError) as error:
            skipped += 1
            print(f"Skipping {source_path}: {error}")
            continue

        processed_records.append(
            {
                "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "original_path": record["path"],
                "surface": record["surface"],
                "label": record["label"],
                "width": str(IMAGE_SIZE[0]),
                "height": str(IMAGE_SIZE[1]),
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "original_path", "surface", "label", "width", "height"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_records)

    print(f"{split_name}: processed={len(processed_records)}, skipped={skipped}")
    print(f"Metadata: {output_csv}")


def main() -> None:
    process_split("train")
    process_split("validation")


if __name__ == "__main__":
    main()
