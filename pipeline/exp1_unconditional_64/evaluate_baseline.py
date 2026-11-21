"""Evaluate baseline samples with lightweight diagnostic proxies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist, pdist


def load_images(paths: list[Path]) -> np.ndarray:
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    if not arrays:
        raise RuntimeError("No images found for evaluation.")
    return np.stack(arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True)
    parser.add_argument("--real-csv", default="data/processed/validation.csv")
    parser.add_argument("--real-count", type=int, default=256)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    generated_path = project_root / args.generated
    real_csv_path = project_root / args.real_csv
    generated_paths = [generated_path] if generated_path.is_file() else sorted(generated_path.glob("*.png"))
    generated = load_images(generated_paths)

    with real_csv_path.open("r", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))[: args.real_count]
    real = load_images([project_root / record["path"] for record in records])

    generated_vectors = generated.reshape(len(generated), -1)
    real_vectors = real.reshape(len(real), -1)
    generated_distances = pdist(generated_vectors, metric="euclidean")
    nearest_real = cdist(generated_vectors, real_vectors, metric="euclidean").min(axis=1)

    print(f"Generated images: {len(generated)}")
    print(f"Real comparison images: {len(real)}")
    print(f"Generated mean pixel value: {generated.mean():.4f}")
    print(f"Real mean pixel value: {real.mean():.4f}")
    print(f"Generated pixel std: {generated.std():.4f}")
    print(f"Real pixel std: {real.std():.4f}")
    print(f"Mean pairwise generated distance: {generated_distances.mean():.4f}")
    print(f"Mean nearest-real pixel distance: {nearest_real.mean():.4f}")
    print("These are lightweight diagnostic proxies, not standard FID/KID/LPIPS.")


if __name__ == "__main__":
    main()
