"""Build and preprocess the Experiment 3 128x128 training dataset."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "processed_128" / "train"
OUTPUT_CSV = ROOT / "data" / "processed_128" / "train.csv"
SDNET_CSV = ROOT / "data" / "splits" / "train.csv"
EXTERNAL_ROOT = ROOT / "data" / "raw" / "external_cracks" / "Reinforced concrete structure crack detection data" / "images"


def make_name(source: str, path: Path) -> str:
    key = f"{source}:{path.as_posix()}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16] + ".png"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    with SDNET_CSV.open("r", encoding="utf-8") as handle:
        sdnet_records = list(csv.DictReader(handle))

    sources = []
    for record in sdnet_records:
        sources.append(
            {
                "source": "SDNET2018",
                "original_path": record["path"],
                "source_path": str(ROOT / record["path"]),
                "surface": record["surface"],
                "label": record["label"],
                "annotation_available": "0",
            }
        )

    external_files = sorted(EXTERNAL_ROOT.glob("*.jpg"))
    for path in external_files:
        sources.append(
            {
                "source": "RC_10519",
                "original_path": path.relative_to(ROOT).as_posix(),
                "source_path": str(path),
                "surface": "reinforced_concrete_unknown",
                "label": "cracked",
                "annotation_available": "1" if (ROOT / "data/raw/external_cracks/Reinforced concrete structure crack detection data/label_json" / f"{path.stem}.json").exists() else "0",
            }
        )

    for item in tqdm(sources, desc="Preprocessing Experiment 3"):
        source_path = Path(item["source_path"])
        output_name = make_name(item["source"], Path(item["original_path"]))
        output_path = OUTPUT_ROOT / output_name
        with Image.open(source_path) as image:
            image.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS).save(output_path, "PNG", optimize=True)
        records.append(
            {
                "path": output_path.relative_to(ROOT).as_posix(),
                "original_path": item["original_path"],
                "source": item["source"],
                "surface": item["surface"],
                "label": item["label"],
                "annotation_available": item["annotation_available"],
                "width": "128",
                "height": "128",
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["path", "original_path", "source", "surface", "label", "annotation_available", "width", "height"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"SDNET2018 records: {len(sdnet_records)}")
    print(f"External cracked records: {len(external_files)}")
    print(f"Combined records: {len(records)}")
    print(f"Output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
