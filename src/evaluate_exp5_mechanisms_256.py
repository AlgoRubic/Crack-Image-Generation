"""
Quantitative Structural Mechanism & Fidelity Evaluation for Experiment 5 (Level B at Native 256x256).
Computes:
1. Per-class structural mechanics metrics:
   - Class 0 (non_cracked): Clean Surface Rate (%)
   - Class 1 (flexural): Vertical Orientation Rate (%), Bottom-to-Top Width Taper Ratio (w_lower / w_upper), L_max, C_ratio
   - Class 2 (flexure_shear): Piecewise Curvature Bend Delta_theta (deg), L_max, C_ratio
   - Class 3 (web_shear): Diagonal Orientation Rate (%), L_max, C_ratio
   - Class 4 (torsion): Parallel Echelon Multi-Strand Rate (%), Mean Diagonal Components, L_max
   - Class 5 (corrosion_splitting): Horizontal Rebar Alignment Rate (%), L_max, C_ratio
   - Class 6 (shrinkage_map): Multi-Branch Crazing Network Rate (%), Mean Significant Components, C_ratio
2. Class-matched Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) via torch_fidelity.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from PIL import Image
from scipy.ndimage import binary_opening, distance_transform_edt, find_objects, label, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_experiment5_mechanisms import (
    CLASS_NAMES,
    angular_diff,
    compute_angle_deg,
    is_diagonal,
    is_horizontal,
    is_vertical,
)

GEN_DIR = ROOT / "experiments" / "structural_mechanism_ddpm" / "generated_samples"
REAL_CSV = ROOT / "data" / "processed_exp5_mechanisms" / "train.csv"
OUT_JSON = ROOT / "experiments" / "structural_mechanism_ddpm" / "EVALUATION_METRICS.json"


def segment_generated_crack(img_path: Path) -> np.ndarray:
    """
    Extract structural crack binary mask from a generated 256x256 concrete image
    while filtering out small 1-2px aggregate pores.
    """
    im = Image.open(img_path).convert("L")
    arr = np.array(im, dtype=np.float32)
    local_mean = uniform_filter(arr, size=19)
    diff = local_mean - arr
    # Threshold at 22 intensity units below local concrete matrix + 2x2 morphological opening
    raw_mask = diff > 22.0
    cleaned = binary_opening(raw_mask, structure=np.ones((2, 2), dtype=bool))
    return cleaned.astype(np.uint8)


def evaluate_patch_mechanics(mask: np.ndarray) -> dict:
    total_px = int(np.sum(mask > 0))
    if total_px < 40:
        return {
            "has_crack": False,
            "total_px": total_px,
            "max_diag": 0.0,
            "c_ratio": 0.0,
            "sig_comps": 0,
            "diag_comps": 0,
            "theta": 0.0,
            "delta_theta": 0.0,
            "taper_ratio": 1.0,
        }

    labeled, n_feat = label(mask > 0)
    slices = find_objects(labeled)
    comps = []
    for idx in range(n_feat):
        sl = slices[idx]
        if sl is None:
            continue
        cm = labeled == (idx + 1)
        sz = int(np.sum(cm))
        dy = sl[0].stop - sl[0].start
        dx = sl[1].stop - sl[1].start
        diag = float(np.sqrt(dx * dx + dy * dy))
        if sz >= 20 and diag >= 18.0:
            comps.append((sz, diag, sl, cm))

    if not comps:
        return {
            "has_crack": False,
            "total_px": total_px,
            "max_diag": 0.0,
            "c_ratio": 0.0,
            "sig_comps": 0,
            "diag_comps": 0,
            "theta": 0.0,
            "delta_theta": 0.0,
            "taper_ratio": 1.0,
        }

    comps.sort(key=lambda x: x[0], reverse=True)
    max_sz, max_diag, max_sl, main_cm = comps[0]
    sig_total = sum(c[0] for c in comps)
    c_ratio = float(max_sz / max(1, sig_total))

    theta_main, _ = compute_angle_deg(main_cm)

    # Piecewise curvature
    y0, y1 = max_sl[0].start, max_sl[0].stop
    ym = (y0 + y1) // 2
    low_m = np.zeros_like(main_cm)
    up_m = np.zeros_like(main_cm)
    low_m[ym:y1, :] = main_cm[ym:y1, :]
    up_m[y0:ym, :] = main_cm[y0:ym, :]
    th_low, _ = compute_angle_deg(low_m)
    th_up, _ = compute_angle_deg(up_m)
    delta_theta = angular_diff(th_low, th_up)

    # EDT taper
    edt = distance_transform_edt(main_cm)
    w_low = float(np.percentile(edt[low_m], 80)) if np.any(low_m) else 1.0
    w_up = float(np.percentile(edt[up_m], 80)) if np.any(up_m) else 1.0
    taper_ratio = w_low / max(0.5, w_up)

    diag_c = sum(1 for c in comps if c[1] >= 30.0 and is_diagonal(compute_angle_deg(c[3])[0], tol=28.0))

    return {
        "has_crack": max_diag >= 40.0,
        "total_px": sig_total,
        "max_diag": max_diag,
        "c_ratio": c_ratio,
        "sig_comps": len(comps),
        "diag_comps": diag_c,
        "theta": theta_main,
        "delta_theta": delta_theta,
        "taper_ratio": taper_ratio,
    }


def compute_fid_kid() -> dict:
    flat_gen = ROOT / "experiments" / "structural_mechanism_ddpm" / "_tmp_flat_gen"
    flat_ref = ROOT / "experiments" / "structural_mechanism_ddpm" / "_tmp_flat_ref"
    flat_gen.mkdir(parents=True, exist_ok=True)
    flat_ref.mkdir(parents=True, exist_ok=True)

    by_class = {cid: [] for cid in CLASS_NAMES}
    with REAL_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = int(r["class_id"])
            if cid in by_class and len(by_class[cid]) < 16:
                p = ROOT / r["path"]
                if p.exists():
                    by_class[cid].append(p)

    for cid, cname in CLASS_NAMES.items():
        for p in sorted((GEN_DIR / cname).glob("sample_*.png")):
            shutil.copy(p, flat_gen / f"{cname}_{p.name}")
        for idx, p in enumerate(by_class[cid]):
            shutil.copy(p, flat_ref / f"{cname}_ref_{idx:03d}.png")

    metrics = torch_fidelity.calculate_metrics(
        input1=str(flat_gen),
        input2=str(flat_ref),
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=True,
        kid_subsets=8,
        kid_subset_size=12,
        verbose=False,
    )

    shutil.rmtree(flat_gen, ignore_errors=True)
    shutil.rmtree(flat_ref, ignore_errors=True)

    return {
        "frechet_inception_distance": round(float(metrics["frechet_inception_distance"]), 3),
        "kernel_inception_distance_mean": round(float(metrics["kernel_inception_distance_mean"]), 5),
        "kernel_inception_distance_std": round(float(metrics["kernel_inception_distance_std"]), 5),
    }


def main() -> None:
    report = {"classes": {}, "fidelity": {}}
    print("--- Evaluating Experiment 5 (Level B: 7 Structural Failure Mechanisms) ---\n")

    for cid, cname in CLASS_NAMES.items():
        cdir = GEN_DIR / cname
        imgs = sorted(cdir.glob("sample_*.png"))
        if not imgs:
            continue

        stats = [evaluate_patch_mechanics(segment_generated_crack(p)) for p in imgs]
        n = len(stats)

        if cid == 0:
            clean = sum(1 for s in stats if not s["has_crack"])
            report["classes"][cname] = {
                "samples": n,
                "clean_surface_rate": round(clean / n, 4),
                "false_positive_rate": round((n - clean) / n, 4),
            }
            print(f"Class 0 ({cname}): Clean Surface Rate = {clean}/{n} ({clean/n*100:.1f}%)")
        else:
            cracked = [s for s in stats if s["has_crack"]] or stats
            if cid == 1:
                adhere = sum(1 for s in stats if s["has_crack"] and is_vertical(s["theta"], tol=28.0))
            elif cid == 2:
                adhere = sum(1 for s in stats if s["has_crack"] and s["delta_theta"] >= 14.0)
            elif cid == 3:
                adhere = sum(1 for s in stats if s["has_crack"] and is_diagonal(s["theta"], tol=28.0))
            elif cid == 4:
                adhere = sum(1 for s in stats if s["diag_comps"] >= 2 or s["sig_comps"] >= 2)
            elif cid == 5:
                adhere = sum(1 for s in stats if s["has_crack"] and is_horizontal(s["theta"], tol=28.0))
            else:
                adhere = sum(1 for s in stats if s["sig_comps"] >= 3 and s["c_ratio"] < 0.65)

            report["classes"][cname] = {
                "samples": n,
                "crack_presence_rate": round(sum(1 for s in stats if s["has_crack"]) / n, 4),
                "mechanism_adherence_rate": round(adhere / n, 4),
                "mean_max_length_px": round(float(np.mean([s["max_diag"] for s in cracked])), 1),
                "mean_continuity_ratio": round(float(np.mean([s["c_ratio"] for s in cracked])), 3),
                "mean_curvature_deg": round(float(np.mean([s["delta_theta"] for s in cracked])), 1),
                "mean_taper_ratio": round(float(np.mean([s["taper_ratio"] for s in cracked])), 2),
                "mean_sig_components": round(float(np.mean([s["sig_comps"] for s in cracked])), 2),
            }
            print(
                f"Class {cid} ({cname:20s}): Presence={report['classes'][cname]['crack_presence_rate']*100:5.1f}%, "
                f"Adherence={report['classes'][cname]['mechanism_adherence_rate']*100:5.1f}%, "
                f"L_max={report['classes'][cname]['mean_max_length_px']:5.1f}px, "
                f"C_ratio={report['classes'][cname]['mean_continuity_ratio']:.3f}, "
                f"Curvature={report['classes'][cname]['mean_curvature_deg']:4.1f}deg, "
                f"Taper={report['classes'][cname]['mean_taper_ratio']:.2f}"
            )

    print("\nComputing Class-Matched FID and KID...")
    report["fidelity"] = compute_fid_kid()
    print(f"FID: {report['fidelity']['frechet_inception_distance']}")
    print(
        f"KID: {report['fidelity']['kernel_inception_distance_mean']} +/- "
        f"{report['fidelity']['kernel_inception_distance_std']}"
    )

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Saved evaluation metrics to: {OUT_JSON}")


if __name__ == "__main__":
    main()
