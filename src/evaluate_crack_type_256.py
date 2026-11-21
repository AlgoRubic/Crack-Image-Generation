"""
Quantitative Evaluation Script for Experiment 4 (Level A Crack-Type DDPM at 256x256).
Computes:
1. Crack Presence Rate per class
2. Morphological Continuity Ratio (C_ratio)
3. Maximum Continuous Crack Length (L_max)
4. Connected Component Count (CC_count)
5. Morphological Orientation Alignment Accuracy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "experiments" / "crack_type_conditional" / "generated_samples"
OUT_JSON = ROOT / "experiments" / "crack_type_conditional" / "EVALUATION_METRICS.json"

CLASSES = [
    (0, "non_cracked"),
    (1, "longitudinal"),
    (2, "transverse"),
    (3, "diagonal"),
    (4, "alligator_map"),
]


def extract_crack_mask(img_path: Path) -> np.ndarray:
    im = Image.open(img_path).convert("L")
    arr = np.array(im, dtype=np.float32)
    # Local adaptive background subtraction
    local_mean = uniform_filter(arr, size=15)
    diff = local_mean - arr
    # Cracks are darker than local concrete background
    mask = (diff > 16.0).astype(np.uint8)
    return mask


def analyze_crack_morphology(mask: np.ndarray):
    total_pixels = int(np.sum(mask > 0))
    if total_pixels < 30:
        return {
            "has_crack": False,
            "total_pixels": total_pixels,
            "max_length": 0.0,
            "c_ratio": 0.0,
            "cc_count": 0,
            "angle": 0.0,
            "predicted_class": "non_cracked",
        }

    labeled, num_features = label(mask > 0)
    if num_features == 0:
        return {
            "has_crack": False,
            "total_pixels": total_pixels,
            "max_length": 0.0,
            "c_ratio": 0.0,
            "cc_count": 0,
            "angle": 0.0,
            "predicted_class": "non_cracked",
        }

    slices = find_objects(labeled)
    comp_sizes = [int(np.sum(labeled == i + 1)) for i in range(num_features)]
    max_idx = int(np.argmax(comp_sizes))
    max_slice = slices[max_idx]

    dy = max_slice[0].stop - max_slice[0].start
    dx = max_slice[1].stop - max_slice[1].start
    diag_len = float(np.sqrt(dx**2 + dy**2))
    c_ratio = float(comp_sizes[max_idx] / (total_pixels + 1e-5))

    # Alligator check: multiple significant components and mesh-like dispersion
    sig_components = sum(1 for sz in comp_sizes if sz >= 25)
    if sig_components >= 4 and c_ratio < 0.50:
        return {
            "has_crack": True,
            "total_pixels": total_pixels,
            "max_length": diag_len,
            "c_ratio": c_ratio,
            "cc_count": num_features,
            "sig_cc_count": sig_components,
            "angle": 0.0,
            "predicted_class": "alligator_map",
        }

    # Orientation analysis via central moments
    main_mask = labeled == (max_idx + 1)
    y, x = np.where(main_mask)
    if len(x) > 5:
        xc, yc = x.mean(), y.mean()
        u20 = np.mean((x - xc) ** 2)
        u02 = np.mean((y - yc) ** 2)
        u11 = np.mean((x - xc) * (y - yc))
        theta_rad = 0.5 * np.arctan2(2 * u11, u20 - u02)
        theta_deg = float(np.degrees(theta_rad))
        if theta_deg < 0:
            theta_deg += 180.0
    else:
        theta_deg = 0.0

    if theta_deg <= 22.5 or theta_deg >= 157.5:
        pred_class = "longitudinal"
    elif 67.5 <= theta_deg <= 112.5:
        pred_class = "transverse"
    else:
        pred_class = "diagonal"

    return {
        "has_crack": diag_len >= 35.0,
        "total_pixels": total_pixels,
        "max_length": diag_len,
        "c_ratio": c_ratio,
        "cc_count": num_features,
        "sig_cc_count": sig_components,
        "angle": theta_deg,
        "predicted_class": pred_class,
    }


def main():
    results = {}
    print("--- Evaluating Experiment 4 Generated Samples (256x256 DDIM 50-step CFG 2.5) ---\n")

    for cid, cname in CLASSES:
        cdir = SAMPLE_DIR / cname
        if not cdir.exists():
            continue
        imgs = sorted(cdir.glob("sample_*.png"))
        if not imgs:
            continue

        class_stats = []
        for im_p in imgs:
            mask = extract_crack_mask(im_p)
            stat = analyze_crack_morphology(mask)
            class_stats.append(stat)

        n = len(class_stats)
        if cname == "non_cracked":
            clean_count = sum(1 for s in class_stats if not s["has_crack"])
            results[cname] = {
                "sample_count": n,
                "clean_surface_rate": round(clean_count / n, 4),
                "false_positive_crack_rate": round((n - clean_count) / n, 4),
                "mean_noise_pixels": round(float(np.mean([s["total_pixels"] for s in class_stats])), 1),
            }
            print(f"Class 0 (non_cracked): Clean Surface Rate = {clean_count}/{n} ({results[cname]['clean_surface_rate']*100:.1f}%)")

        elif cname == "alligator_map":
            detected_mesh = sum(1 for s in class_stats if s["predicted_class"] == "alligator_map" or s["sig_cc_count"] >= 3)
            mean_sig_cc = float(np.mean([s["sig_cc_count"] for s in class_stats]))
            mean_c_ratio = float(np.mean([s["c_ratio"] for s in class_stats]))
            results[cname] = {
                "sample_count": n,
                "mesh_detection_rate": round(detected_mesh / n, 4),
                "mean_significant_components": round(mean_sig_cc, 2),
                "mean_c_ratio": round(mean_c_ratio, 4),
                "mean_total_crack_pixels": round(float(np.mean([s["total_pixels"] for s in class_stats])), 1),
            }
            print(f"Class 4 (alligator_map): Mesh Detection = {detected_mesh}/{n} ({results[cname]['mesh_detection_rate']*100:.1f}%), Mean Sig Components = {mean_sig_cc:.1f}")

        else:
            crack_presence = sum(1 for s in class_stats if s["has_crack"])
            alignment_count = sum(1 for s in class_stats if s["predicted_class"] == cname)
            mean_l_max = float(np.mean([s["max_length"] for s in class_stats]))
            mean_c_ratio = float(np.mean([s["c_ratio"] for s in class_stats]))
            mean_cc_count = float(np.mean([s["sig_cc_count"] for s in class_stats]))

            results[cname] = {
                "sample_count": n,
                "crack_presence_rate": round(crack_presence / n, 4),
                "orientation_alignment_accuracy": round(alignment_count / n, 4),
                "mean_max_length_px": round(mean_l_max, 1),
                "mean_continuity_ratio": round(mean_c_ratio, 4),
                "mean_significant_components": round(mean_cc_count, 2),
            }
            print(f"Class {cid} ({cname}): Crack Presence = {crack_presence}/{n} ({results[cname]['crack_presence_rate']*100:.1f}%), Alignment = {alignment_count}/{n} ({results[cname]['orientation_alignment_accuracy']*100:.1f}%), Mean L_max = {mean_l_max:.1f}px, Mean C_ratio = {mean_c_ratio:.3f}")

    # Baseline comparison (Old 128px upscaled model metrics from CRACK_CONTINUITY_ANALYSIS_REPORT.md)
    results["baseline_comparison"] = {
        "exp3_128px_upscaled": {
            "resolution": "128x128 -> 256x256 upscaled",
            "mean_continuity_ratio": 0.284,
            "mean_connected_components": 14.2,
            "crack_fragmentation": "Severe (isolated specks/hairline blur)",
        },
        "exp4_256px_attention": {
            "resolution": "256x256 Native AttentionUNet",
            "mean_linear_continuity_ratio": round(float(np.mean([results[c]["mean_continuity_ratio"] for c in ["longitudinal", "transverse", "diagonal"]])), 3),
            "mean_linear_components": round(float(np.mean([results[c]["mean_significant_components"] for c in ["longitudinal", "transverse", "diagonal"]])), 2),
            "crack_fragmentation": "Not established by this detector; independent validation required",
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Quantitative metrics saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
