"""
Class-matched FID and KID evaluation for Experiment 4 (Native 256x256).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch
import torch_fidelity

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "experiments" / "crack_type_conditional" / "generated_samples"
REAL_CSV = ROOT / "data" / "processed_256" / "train.csv"
REF_DIR = ROOT / "experiments" / "crack_type_conditional" / "reference_samples"
OUT_JSON = ROOT / "experiments" / "crack_type_conditional" / "FID_KID_METRICS.json"

CLASSES = ["non_cracked", "longitudinal", "transverse", "diagonal", "alligator_map"]


def prepare_reference_subset(count_per_class: int = 16):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    by_class = {c: [] for c in CLASSES}
    with open(REAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            c = r["class_name"]
            if c in by_class and len(by_class[c]) < count_per_class:
                src_p = ROOT / r["path"]
                if src_p.exists():
                    by_class[c].append(src_p)

    # Flatten reference images into one folder
    for c, paths in by_class.items():
        c_ref_dir = REF_DIR / c
        c_ref_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, c_ref_dir / p.name)

    print(f"Prepared reference set with {sum(len(v) for v in by_class.values())} real images.")


def prepare_flattened_gen_dir():
    flat_gen = ROOT / "experiments" / "crack_type_conditional" / "flat_generated"
    flat_ref = ROOT / "experiments" / "crack_type_conditional" / "flat_reference"
    flat_gen.mkdir(parents=True, exist_ok=True)
    flat_ref.mkdir(parents=True, exist_ok=True)

    for c in CLASSES:
        for p in (GEN_DIR / c).glob("sample_*.png"):
            shutil.copy(p, flat_gen / f"{c}_{p.name}")
        for p in (REF_DIR / c).glob("*.png"):
            shutil.copy(p, flat_ref / f"{c}_{p.name}")

    return flat_gen, flat_ref


def main():
    print("Preparing reference subset...")
    prepare_reference_subset(count_per_class=16)
    flat_gen, flat_ref = prepare_flattened_gen_dir()

    print("Computing Inception Metrics (FID, KID) with torch_fidelity...")
    metrics = torch_fidelity.calculate_metrics(
        input1=str(flat_gen),
        input2=str(flat_ref),
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=True,
        kid_subsets=8,
        kid_subset_size=10,
        verbose=False,
    )

    results = {
        "frechet_inception_distance": round(float(metrics["frechet_inception_distance"]), 3),
        "kernel_inception_distance_mean": round(float(metrics["kernel_inception_distance_mean"]), 5),
        "kernel_inception_distance_std": round(float(metrics["kernel_inception_distance_std"]), 5),
        "sample_count": len(list(flat_gen.glob("*.png"))),
        "reference_count": len(list(flat_ref.glob("*.png"))),
    }

    print("\n--- Quantitative Fidelity Metrics ---")
    print(f"FID: {results['frechet_inception_distance']}")
    print(f"KID: {results['kernel_inception_distance_mean']} +/- {results['kernel_inception_distance_std']}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved fidelity metrics to: {OUT_JSON}")


if __name__ == "__main__":
    main()
