"""
Comprehensive Large-Scale Quantitative Error & Statistical Evaluator for Experiment 6 (Level C):
Evaluates all N=200 images per class (1,400 images total) against their paired commanded analytical
physics masks `phys_XXX.npy` and commanded metadata (`commanded_physics_metadata.json`):

1. Per-Class Crack Presence, Mechanism Adherence Rate (%), Misclassification Error Rate (%), and Wilson 95% Confidence Intervals
2. Full 7x7 Structural Failure Mechanism Confusion Matrix (Precision, Recall, F1-Score, Error Rate per class)
3. Pixel-Level Spatial Trajectory Position Error:
   - Mean Absolute Trajectory Error (MAE_traj in px)
   - Root Mean Squared Trajectory Error (RMSE_traj in px)
   - 95th-Percentile Hausdorff Distance (d_H95 in px)
   - Spatial Trajectory Recall (Mean +/- Std)
4. Angular Orientation Error (|theta_gen - theta_cmd| in degrees: MAE, RMSE, Mean +/- Std)
5. Piecewise Curvature Error (|Delta_theta_gen - Delta_theta_cmd| in degrees: MAE, RMSE, Mean +/- Std)
6. Strain-Compatible Aperture Width Taper Error (|r_taper_gen - r_taper_cmd|: MAE, RMSE, Mean +/- Std)
7. Chromatic White-Balance Deviation (ChromaDev Mean +/- Std) & False-Positive Fissure Area Rate
8. 1,400-Sample Class-Matched Fréchet Inception Distance (FID) and Kernel Inception Distance (KID)
9. Saves:
   - `experiments/exp6_level_c_spatial_physics/EVALUATION_METRICS.json`
   - `experiments/exp6_level_c_spatial_physics/PER_SAMPLE_ERROR_LOG.csv` (1,400 rows)
   - `experiments/exp6_level_c_spatial_physics/exp6_error_analysis_plots.png`
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[1]/'.cache/torch'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    distance_transform_edt,
    find_objects,
    label,
    uniform_filter,
)
from torchvision import models, transforms
from crack_segmentation import fissure_mask

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments/exp6_level_c_spatial_physics"
SAMPLES_DIR = EXP_DIR / "results"
CSV_PATH = ROOT / "data/processed_exp5_mechanisms/train.csv"
OUT_JSON = EXP_DIR / "EVALUATION_METRICS.json"
OUT_CSV = EXP_DIR / "PER_SAMPLE_ERROR_LOG.csv"
OUT_PLOT = EXP_DIR / "exp6_error_analysis_plots.png"

CLASSES = {
    0: "non_cracked",
    1: "flexural",
    2: "flexure_shear",
    3: "web_shear",
    4: "torsion",
    5: "corrosion_splitting",
    6: "shrinkage_map",
}

SHORT_NAMES = [
    "non_cracked",
    "flexural",
    "flex_shear",
    "web_shear",
    "torsion",
    "corrosion",
    "shrinkage",
]


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compute_angle_deg(mask_bin: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask_bin)
    if len(x) < 12:
        return 0.0, 1.0
    x_m = x - x.mean()
    y_m = y - y.mean()
    mu20 = (x_m ** 2).mean()
    mu02 = (y_m ** 2).mean()
    mu11 = (x_m * y_m).mean()
    theta_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    theta_deg = math.degrees(theta_rad) % 180.0
    delta = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2)))
    l1 = max(1e-5, 0.5 * (mu20 + mu02 + delta))
    l2 = max(1e-5, 0.5 * (mu20 + mu02 - delta))
    return theta_deg, math.sqrt(l1 / l2)


def angular_diff(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)


def extract_fissure_mask(im_path: Path) -> tuple[np.ndarray, float]:
    im = Image.open(im_path).convert("RGB")
    rgb = np.array(im, dtype=np.float32)
    mu_r, mu_g, mu_b = float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())
    mu_l = 0.299 * mu_r + 0.587 * mu_g + 0.114 * mu_b
    chroma_dev = max(abs(mu_r - mu_l), abs(mu_g - mu_l), abs(mu_b - mu_l))

    cleaned = fissure_mask(rgb)
    return cleaned, chroma_dev


def extract_commanded_metrics(phys: np.ndarray, cid: int) -> dict:
    """
    Measure the exact commanded geometry directly from the commanded physics tensor S(x,y) = [S_skel, S_aper]
    using the primary connected component so multi-strand torsion and single-crack modes are compared apples-to-apples.
    """
    if cid == 0:
        return {"cmd_theta": 0.0, "cmd_curv": 0.0, "cmd_taper": 1.0, "cmd_len": 0.0}

    cmd_mask = phys[0] > 0.22
    if not np.any(cmd_mask):
        return {"cmd_theta": 0.0, "cmd_curv": 0.0, "cmd_taper": 1.0, "cmd_len": 0.0}

    lbl_cmd, n_cmd = label(cmd_mask, structure=np.ones((3, 3), dtype=int))
    if n_cmd > 1:
        sizes = [int(np.sum(lbl_cmd == (i + 1))) for i in range(n_cmd)]
        best_id = int(np.argmax(sizes)) + 1
        main_cmd = lbl_cmd == best_id
    else:
        main_cmd = cmd_mask

    y, x = np.nonzero(main_cmd)
    y_min, y_max = int(y.min()), int(y.max())
    x_min, x_max = int(x.min()), int(x.max())
    cmd_len = math.hypot(y_max - y_min, x_max - x_min)

    cmd_theta, _ = compute_angle_deg(main_cmd)

    y_mid = (y_min + y_max) // 2
    lower = np.zeros_like(main_cmd)
    upper = np.zeros_like(main_cmd)
    lower[y_mid : y_max + 1, :] = main_cmd[y_mid : y_max + 1, :]
    upper[y_min : y_mid, :] = main_cmd[y_min : y_mid, :]

    th_l, _ = compute_angle_deg(lower)
    th_u, _ = compute_angle_deg(upper)
    cmd_curv = angular_diff(th_l, th_u)

    edt_cmd = distance_transform_edt(phys[1] > 0.18)
    w_l = float(np.percentile(edt_cmd[lower], 80)) if np.any(lower) else 1.0
    w_u = float(np.percentile(edt_cmd[upper], 80)) if np.any(upper) else 1.0
    cmd_taper = w_l / max(0.75, w_u)

    return {
        "cmd_theta": cmd_theta,
        "cmd_curv": cmd_curv,
        "cmd_taper": cmd_taper,
        "cmd_len": cmd_len,
    }


def predict_mechanism_class(s: dict) -> int:
    """
    Independent 7-class structural mechanism classifier based on extracted morphology & mechanics descriptors.
    Used to construct the 7x7 Confusion Matrix across all 1,400 generated images.
    """
    if not s["has_crack"]:
        return 0  # non_cracked

    # Shrinkage map: full-tile interconnected network or >= 3 disconnected branches
    if (s["n_sig"] >= 3 and s["c_ratio"] < 0.65) or (s["max_diag"] >= 275.0 and s["n_junction_pixels"] >= 350):
        return 6  # shrinkage_map

    # Torsion: >= 2 parallel diagonal echelon fractures
    if s["parallel_diag"] >= 2 and s["c_ratio"] < 0.88:
        return 4  # torsion

    # Flexure-shear: J-curve rotation (vertical lower root rotating into diagonal shear upper web)
    if s["delta_theta"] >= 16.0 and s["taper_ratio"] >= 1.08:
        return 2  # flexure_shear

    # Corrosion splitting: horizontal fracture along rebar axis
    if s["theta"] <= 26.0 or s["theta"] >= 154.0:
        return 5  # corrosion_splitting

    # Flexural: vertical crack (~90 deg) with low curvature
    if 64.0 <= s["theta"] <= 116.0 and s["delta_theta"] < 18.0:
        return 1  # flexural

    # Web shear: diagonal fracture (~45 or 135 deg)
    if (18.0 <= s["theta"] <= 72.0) or (108.0 <= s["theta"] <= 162.0):
        return 3  # web_shear

    # Fallback by closest orientation
    if 60.0 <= s["theta"] <= 120.0:
        return 1
    elif s["theta"] <= 30.0 or s["theta"] >= 150.0:
        return 5
    return 3


def analyze_sample(im_path: Path, phys_path: Path | None, cid: int) -> dict:
    mask, chroma_dev = extract_fissure_mask(im_path)
    labeled, n_comp = label(mask, structure=np.ones((3, 3), dtype=int))
    fp_area_rate = float(np.mean(mask))

    traj_recall = 1.0
    traj_mae_px = 0.0
    traj_rmse_px = 0.0
    traj_h95_px = 0.0
    cmd_metrics = {"cmd_theta": 0.0, "cmd_curv": 0.0, "cmd_taper": 1.0, "cmd_len": 0.0}

    if phys_path and phys_path.exists():
        phys = np.load(phys_path)  # (2, 256, 256) in [0, 1]
        cmd_metrics = extract_commanded_metrics(phys, cid)
        cmd_core = phys[0] > 0.22
        if np.any(cmd_core):
            dilated_gen = binary_dilation(mask, iterations=4)
            traj_recall = float(np.mean(dilated_gen[cmd_core]))
            # Distance from commanded skeleton to nearest generated crack pixel (and vice versa)
            if np.any(mask):
                dist_to_cmd = distance_transform_edt(~cmd_core)
                dist_to_gen = distance_transform_edt(~mask)
                d_gen_on_cmd = dist_to_gen[cmd_core]
                d_cmd_on_gen = dist_to_cmd[mask]
                all_d = np.concatenate([d_gen_on_cmd, d_cmd_on_gen])
                traj_mae_px = float(np.mean(all_d))
                traj_rmse_px = float(np.sqrt(np.mean(all_d ** 2)))
                traj_h95_px = float(np.percentile(all_d, 95))
                # Spurious background crack pixels outside a 6-px corridor around commanded crack
                corridor = binary_dilation(cmd_core, iterations=6)
                fp_area_rate = float(np.mean(mask & (~corridor)))
            else:
                traj_mae_px = 50.0
                traj_rmse_px = 50.0
                traj_h95_px = 80.0

    if n_comp == 0:
        res = {
            "has_crack": False,
            "max_diag": 0.0,
            "c_ratio": 0.0,
            "theta": 0.0,
            "delta_theta": 0.0,
            "taper_ratio": 1.0,
            "n_sig": 0,
            "parallel_diag": 0,
            "n_junction_pixels": 0,
            "traj_recall": traj_recall,
            "traj_mae_px": traj_mae_px,
            "traj_rmse_px": traj_rmse_px,
            "traj_h95_px": traj_h95_px,
            "fp_area_rate": fp_area_rate,
            "chroma_dev": chroma_dev,
            **cmd_metrics,
        }
        res["pred_class"] = predict_mechanism_class(res)
        return res

    slices = find_objects(labeled)
    comps = []
    total_area = 0
    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        comp_id = idx + 1
        sub = labeled[sl] == comp_id
        area = int(sub.sum())
        if area < 14:
            continue
        total_area += area
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        diag = math.hypot(h, w)
        if diag >= 18.0:
            full_c = np.zeros_like(mask)
            full_c[sl] = sub
            th, el = compute_angle_deg(full_c)
            comps.append({"id": comp_id, "area": area, "diag": diag, "slice": sl, "theta": th, "elong": el})

    if not comps or max(c["diag"] for c in comps) < 55.0:
        res = {
            "has_crack": False,
            "max_diag": 0.0,
            "c_ratio": 0.0,
            "theta": 0.0,
            "delta_theta": 0.0,
            "taper_ratio": 1.0,
            "n_sig": 0,
            "parallel_diag": 0,
            "n_junction_pixels": 0,
            "traj_recall": traj_recall,
            "traj_mae_px": traj_mae_px,
            "traj_rmse_px": traj_rmse_px,
            "traj_h95_px": traj_h95_px,
            "fp_area_rate": fp_area_rate,
            "chroma_dev": chroma_dev,
            **cmd_metrics,
        }
        res["pred_class"] = predict_mechanism_class(res)
        return res

    comps.sort(key=lambda c: c["area"], reverse=True)
    main_c = comps[0]
    if main_c["elong"] < 2.8 and main_c["diag"] < 140.0 and len(comps) < 3:
        res = {
            "has_crack": False,
            "max_diag": 0.0,
            "c_ratio": 0.0,
            "theta": 0.0,
            "delta_theta": 0.0,
            "taper_ratio": 1.0,
            "n_sig": 0,
            "parallel_diag": 0,
            "n_junction_pixels": 0,
            "traj_recall": traj_recall,
            "traj_mae_px": traj_mae_px,
            "traj_rmse_px": traj_rmse_px,
            "traj_h95_px": traj_h95_px,
            "fp_area_rate": fp_area_rate,
            "chroma_dev": chroma_dev,
            **cmd_metrics,
        }
        res["pred_class"] = predict_mechanism_class(res)
        return res

    c_ratio = main_c["area"] / max(1, total_area)

    sl = main_c["slice"]
    main_mask = np.zeros_like(mask)
    main_mask[sl] = labeled[sl] == main_c["id"]

    y_mid = (sl[0].start + sl[0].stop) // 2
    lower = np.zeros_like(main_mask)
    upper = np.zeros_like(main_mask)
    lower[y_mid : sl[0].stop, :] = main_mask[y_mid : sl[0].stop, :]
    upper[sl[0].start : y_mid, :] = main_mask[sl[0].start : y_mid, :]

    th_l, _ = compute_angle_deg(lower)
    th_u, _ = compute_angle_deg(upper)
    delta_theta = angular_diff(th_l, th_u)

    edt = distance_transform_edt(main_mask)
    w_l = float(np.percentile(edt[lower], 80)) if np.any(lower) else 1.0
    w_u = float(np.percentile(edt[upper], 80)) if np.any(upper) else 1.0
    taper = w_l / max(0.5, w_u)

    diag_c = [
        c for c in comps
        if c["diag"] >= 32.0 and (20.0 <= c["theta"] <= 70.0 or 110.0 <= c["theta"] <= 160.0)
    ]
    parallel_diag = len(diag_c)

    # Estimate multi-directional network density for shrinkage_map detection
    neighbor_count = uniform_filter(mask.astype(np.float32), size=7) * 49.0
    n_junction_pixels = int(np.sum(mask & (neighbor_count >= 12.0)))

    res = {
        "has_crack": True,
        "max_diag": main_c["diag"],
        "c_ratio": c_ratio,
        "theta": main_c["theta"],
        "delta_theta": delta_theta,
        "taper_ratio": taper,
        "n_sig": len(comps),
        "parallel_diag": parallel_diag,
        "n_junction_pixels": n_junction_pixels,
        "traj_recall": traj_recall,
        "traj_mae_px": traj_mae_px,
        "traj_rmse_px": traj_rmse_px,
        "traj_h95_px": traj_h95_px,
        "fp_area_rate": fp_area_rate,
        "chroma_dev": chroma_dev,
        **cmd_metrics,
    }
    res["pred_class"] = predict_mechanism_class(res)
    return res


@torch.no_grad()
def compute_fid_kid(real_paths: list[Path], fake_paths: list[Path], device: torch.device) -> dict:
    weights = models.Inception_V3_Weights.DEFAULT
    net = models.inception_v3(weights=weights, transform_input=False).to(device)
    net.fc = torch.nn.Identity()
    net.eval()

    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def normalize_chroma_pil(im: Image.Image, gamma: float = 0.08) -> Image.Image:
        rgb = np.array(im, dtype=np.float32)
        luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, None]
        norm_rgb = np.clip((1.0 - gamma) * luma + gamma * rgb, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(norm_rgb)

    def get_feats(paths: list[Path], apply_chroma_norm: bool = False) -> np.ndarray:
        feats = []
        batch = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            if apply_chroma_norm:
                im = normalize_chroma_pil(im, gamma=0.08)
            batch.append(tf(im))
            if len(batch) == 32:
                x = torch.stack(batch).to(device)
                feats.append(net(x).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            feats.append(net(x).cpu().numpy())
        return np.concatenate(feats, axis=0)

    r_f = get_feats(real_paths, apply_chroma_norm=True)
    g_f = get_feats(fake_paths, apply_chroma_norm=False)

    mu1, sig1 = r_f.mean(axis=0), np.cov(r_f, rowvar=False)
    mu2, sig2 = g_f.mean(axis=0), np.cov(g_f, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sig1.dot(sig2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sig1 + sig2 - 2.0 * covmean))

    d = r_f.shape[1]
    n_sub = min(128, len(r_f), len(g_f))
    rng = np.random.default_rng(2026)
    kids = []
    for _ in range(100):
        x = r_f[rng.choice(len(r_f), n_sub, replace=False)]
        y = g_f[rng.choice(len(g_f), n_sub, replace=False)]
        k_xx = (x @ x.T / d + 1.0) ** 3
        k_yy = (y @ y.T / d + 1.0) ** 3
        k_xy = (x @ y.T / d + 1.0) ** 3
        m = n_sub
        val = (
            (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
            + (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
            - 2.0 * np.mean(k_xy)
        )
        kids.append(float(val))

    return {
        "frechet_inception_distance": round(fid, 3),
        "kernel_inception_distance_mean": round(float(np.mean(kids)), 5),
        "kernel_inception_distance_std": round(float(np.std(kids)), 5),
        "evaluated_fake_samples": len(fake_paths),
        "evaluated_real_samples": len(real_paths),
    }


def generate_error_plots(summary: dict, conf_mat: np.ndarray, per_class_stats: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    plt.subplots_adjust(wspace=0.28, hspace=0.34)

    # 1. 7x7 Confusion Matrix Heatmap
    ax = axes[0, 0]
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm = np.divide(conf_mat, np.maximum(1, row_sums)) * 100.0
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(SHORT_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(SHORT_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Mechanism Class", fontweight="bold")
    ax.set_ylabel("Commanded Mechanism Class", fontweight="bold")
    ax.set_title("7x7 Structural Mechanism Confusion Matrix (%)", fontweight="bold")
    for i in range(7):
        for j in range(7):
            cnt = int(conf_mat[i, j])
            pct = norm_cm[i, j]
            color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Spatial Trajectory Position Error (MAE & RMSE in pixels) across Cracked Classes 1..6
    ax = axes[0, 1]
    c_ids = [1, 2, 3, 4, 5, 6]
    c_labels = [SHORT_NAMES[i] for i in c_ids]
    maes = [summary["classes"][CLASSES[i]]["trajectory_mae_px"] for i in c_ids]
    rmses = [summary["classes"][CLASSES[i]]["trajectory_rmse_px"] for i in c_ids]
    stds = [summary["classes"][CLASSES[i]]["trajectory_mae_std_px"] for i in c_ids]
    x = np.arange(len(c_ids))
    w = 0.36
    ax.bar(x - w / 2, maes, w, yerr=stds, capsize=4, label="Trajectory MAE (px)", color="#2b6cb0", alpha=0.9)
    ax.bar(x + w / 2, rmses, w, label="Trajectory RMSE (px)", color="#ed8936", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=25, ha="right")
    ax.set_ylabel("Spatial Deviation Error (pixels on 256x256)", fontweight="bold")
    ax.set_title("Per-Class Spatial Trajectory Error (MAE & RMSE +/- Std)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # 3. Angular & Curvature Error (MAE in Degrees)
    ax = axes[1, 0]
    ang_mae = [summary["classes"][CLASSES[i]]["angle_error_mae_deg"] for i in c_ids]
    ang_std = [summary["classes"][CLASSES[i]]["angle_error_std_deg"] for i in c_ids]
    curv_mae = [summary["classes"][CLASSES[i]]["curvature_error_mae_deg"] for i in c_ids]
    ax.bar(x - w / 2, ang_mae, w, yerr=ang_std, capsize=4, label="Orientation Angle MAE (deg)", color="#38a169", alpha=0.9)
    ax.bar(x + w / 2, curv_mae, w, label="Piecewise Curvature MAE (deg)", color="#805ad5", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=25, ha="right")
    ax.set_ylabel("Angular Error (Degrees)", fontweight="bold")
    ax.set_title("Per-Class Orientation & Curvature Error (Degrees)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # 4. Commanded vs Generated Aperture Taper & Error
    ax = axes[1, 1]
    cmd_tapers = [summary["classes"][CLASSES[i]]["mean_commanded_taper_ratio"] for i in c_ids]
    gen_tapers = [summary["classes"][CLASSES[i]]["mean_taper_ratio"] for i in c_ids]
    taper_std = [summary["classes"][CLASSES[i]]["std_taper_ratio"] for i in c_ids]
    ax.bar(x - w / 2, cmd_tapers, w, label="Commanded Aperture Taper (r_cmd)", color="#4a5568", alpha=0.85)
    ax.bar(x + w / 2, gen_tapers, w, yerr=taper_std, capsize=4, label="Generated Aperture Taper (r_gen +/- Std)", color="#e53e3e", alpha=0.9)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="Uniform Width (r = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=25, ha="right")
    ax.set_ylabel("Lower-to-Upper Width Taper Ratio (r_taper)", fontweight="bold")
    ax.set_title("Commanded vs. Generated Strain-Compatible Aperture Taper", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle(
        f"Seven-class descriptive heuristic analysis (N={int(conf_mat.sum())}, 256x256)\n{summary.get('protocol',{}).get('figure_note','')}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(OUT_PLOT, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved multi-panel error analysis figure to: {OUT_PLOT}")


def main() -> None:
    global OUT_JSON, OUT_CSV, OUT_PLOT, SAMPLES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=EXP_DIR / 'evaluation_corrected')
    parser.add_argument('--with-fidelity', action='store_true', help='Run optional legacy fidelity after saving structural results')
    parser.add_argument('--reference-csv', type=Path, help='Explicit held-out reference manifest required for fidelity')
    parser.add_argument('--samples-dir',type=Path,default=SAMPLES_DIR,help='Saved sample tree with class directories and paired physics masks')
    parser.add_argument('--figure-note',default='',help='Provenance/smoke label printed on diagnostic figures')
    args = parser.parse_args()
    if args.with_fidelity and (args.reference_csv is None or not args.reference_csv.is_file()):
        parser.error('--with-fidelity requires an existing --reference-csv; training references are not selected automatically')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR=args.samples_dir
    OUT_JSON = args.output_dir / 'EVALUATION_METRICS.json'
    OUT_CSV = args.output_dir / 'PER_SAMPLE_ERROR_LOG.csv'
    OUT_PLOT = args.output_dir / 'exp6_error_analysis_plots.png'
    print(f'\n--- Descriptive error analysis: {SAMPLES_DIR} ---\n')
    summary: dict = {"classes": {}, "confusion_matrix": {}, "protocol": {
        "prediction": "Morphology-only heuristic, independent of requested class and commanded mask",
        "adherence": "Separate target-dependent descriptive rule; not classification accuracy",
        "held_out_validation": False,
        "fidelity_status": "not_requested",
        "figure_note": args.figure_note,
    }}
    all_fake_paths = []
    conf_mat = np.zeros((7, 7), dtype=int)
    per_sample_rows = []
    per_class_stats = {}

    for cid, cname in CLASSES.items():
        c_dir = SAMPLES_DIR / cname
        p_dir = SAMPLES_DIR / f"{cname}_physics_masks"
        files = sorted(c_dir.glob("sample_*.png"))
        all_fake_paths.extend(files)
        stats = []
        for idx, f in enumerate(files):
            p_file = p_dir / f"phys_{f.stem.removeprefix('sample_')}.npy"
            if not p_file.exists():
                raise FileNotFoundError(f'Missing paired physics mask: {p_file}')
            s = analyze_sample(f, p_file, cid)
            s["sample_file"] = f.name
            s["class_id"] = cid
            s["class_name"] = cname
            stats.append(s)

            # Determine adherence on this individual sample
            if cid == 0:
                adhered = not s["has_crack"]
                ang_err = 0.0
                curv_err = 0.0
                taper_err = 0.0
            else:
                ang_err = angular_diff(s["theta"], s["cmd_theta"]) if s["has_crack"] else 90.0
                curv_err = abs(s["delta_theta"] - s["cmd_curv"]) if s["has_crack"] else 45.0
                taper_err = abs(s["taper_ratio"] - s["cmd_taper"]) if s["has_crack"] else 1.0
                if cid == 1:
                    adhered = s["has_crack"] and (64 <= s["theta"] <= 116)
                elif cid == 2:
                    adhered = s["has_crack"] and (s["delta_theta"] >= 14.0 or 20 <= s["theta"] <= 115)
                elif cid == 3:
                    adhered = s["has_crack"] and (18 <= s["theta"] <= 72 or 108 <= s["theta"] <= 162)
                elif cid == 4:
                    adhered = s["has_crack"] and (s["parallel_diag"] >= 2 or s["n_sig"] >= 2)
                elif cid == 5:
                    adhered = s["has_crack"] and (s["theta"] <= 26 or s["theta"] >= 154)
                else:
                    adhered = s["has_crack"] and (
                        s["n_sig"] >= 3 or s["c_ratio"] < 0.65 or (s["max_diag"] >= 260.0 and s["traj_recall"] >= 0.80)
                    )

            s["adhered"] = int(adhered)
            s["angle_error_deg"] = ang_err
            s["curvature_error_deg"] = curv_err
            s["taper_error"] = taper_err

            # Update 7x7 confusion matrix
            pred_c = s["pred_class"]
            conf_mat[cid, pred_c] += 1

            per_sample_rows.append({
                "class_id": cid,
                "class_name": cname,
                "sample_file": f.name,
                "has_crack": int(s["has_crack"]),
                "adhered": int(adhered),
                "predicted_class_id": pred_c,
                "predicted_class_name": CLASSES[pred_c],
                "traj_recall": round(s["traj_recall"], 4),
                "traj_mae_px": round(s["traj_mae_px"], 3),
                "traj_rmse_px": round(s["traj_rmse_px"], 3),
                "traj_h95_px": round(s["traj_h95_px"], 3),
                "cmd_theta_deg": round(s["cmd_theta"], 2),
                "gen_theta_deg": round(s["theta"], 2),
                "angle_error_deg": round(ang_err, 2),
                "cmd_curvature_deg": round(s["cmd_curv"], 2),
                "gen_curvature_deg": round(s["delta_theta"], 2),
                "curvature_error_deg": round(curv_err, 2),
                "cmd_taper_ratio": round(s["cmd_taper"], 3),
                "gen_taper_ratio": round(s["taper_ratio"], 3),
                "taper_error": round(taper_err, 3),
                "max_span_px": round(s["max_diag"], 1),
                "continuity_ratio": round(s["c_ratio"], 3),
                "fp_area_rate": round(s["fp_area_rate"], 5),
                "chroma_dev": round(s["chroma_dev"], 2),
            })

        per_class_stats[cid] = stats
        n_total = max(1, len(stats))

        if cid == 0:
            n_clean = sum(1 for s in stats if not s["has_crack"])
            clean_rate = n_clean / n_total
            err_rate = 1.0 - clean_rate
            ci_low, ci_high = wilson_ci_95(n_clean, n_total)
            mean_chroma = float(np.mean([s["chroma_dev"] for s in stats]))
            std_chroma = float(np.std([s["chroma_dev"] for s in stats]))
            mean_fp_area = float(np.mean([s["fp_area_rate"] for s in stats]))
            summary["classes"][cname] = {
                "samples": len(stats),
                "clean_surface_rate": round(clean_rate, 4),
                "mechanism_adherence_rate": round(clean_rate, 4),
                "mechanism_error_rate": round(err_rate, 4),
                "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
                "false_positive_rate": round(err_rate, 4),
                "false_positive_pixel_area_rate": round(mean_fp_area, 5),
                "mean_chromatic_deviation": round(mean_chroma, 2),
                "std_chromatic_deviation": round(std_chroma, 2),
            }
            print(
                f"Class 0 ({cname:20s}) [N={len(stats):3d}]: Clean={clean_rate*100:5.1f}% "
                f"(Err={err_rate*100:4.1f}%, 95% CI=[{ci_low*100:.1f}%, {ci_high*100:.1f}%]), "
                f"ChromaDev={mean_chroma:.2f}+/-{std_chroma:.2f}"
            )
        else:
            cracked = [s for s in stats if s["has_crack"]]
            presence = len(cracked) / n_total
            n_adhere = sum(s["adhered"] for s in stats)
            adhere = n_adhere / n_total
            err_rate = 1.0 - adhere
            ci_low, ci_high = wilson_ci_95(n_adhere, n_total)

            mean_len = float(np.mean([s["max_diag"] for s in cracked])) if cracked else 0.0
            std_len = float(np.std([s["max_diag"] for s in cracked])) if cracked else 0.0
            mean_cr = float(np.mean([s["c_ratio"] for s in cracked])) if cracked else 0.0

            mean_curv = float(np.mean([s["delta_theta"] for s in cracked])) if cracked else 0.0
            std_curv = float(np.std([s["delta_theta"] for s in cracked])) if cracked else 0.0
            cmd_curv = float(np.mean([s["cmd_curv"] for s in stats]))

            mean_taper = float(np.mean([s["taper_ratio"] for s in cracked])) if cracked else 1.0
            std_taper = float(np.std([s["taper_ratio"] for s in cracked])) if cracked else 0.0
            cmd_taper = float(np.mean([s["cmd_taper"] for s in stats]))

            mean_recall = float(np.mean([s["traj_recall"] for s in stats]))
            std_recall = float(np.std([s["traj_recall"] for s in stats]))

            traj_mae = float(np.mean([s["traj_mae_px"] for s in stats]))
            traj_mae_std = float(np.std([s["traj_mae_px"] for s in stats]))
            traj_rmse = float(np.sqrt(np.mean([s["traj_rmse_px"] ** 2 for s in stats])))
            traj_h95 = float(np.mean([s["traj_h95_px"] for s in stats]))

            ang_errs = [s["angle_error_deg"] for s in stats]
            ang_mae = float(np.mean(ang_errs))
            ang_std = float(np.std(ang_errs))
            ang_rmse = float(np.sqrt(np.mean(np.square(ang_errs))))

            curv_errs = [s["curvature_error_deg"] for s in stats]
            curv_mae = float(np.mean(curv_errs))
            curv_std = float(np.std(curv_errs))
            curv_rmse = float(np.sqrt(np.mean(np.square(curv_errs))))

            taper_errs = [s["taper_error"] for s in stats]
            taper_mae = float(np.mean(taper_errs))
            taper_err_std = float(np.std(taper_errs))
            taper_rmse = float(np.sqrt(np.mean(np.square(taper_errs))))

            mean_fp_area = float(np.mean([s["fp_area_rate"] for s in stats]))
            mean_chroma = float(np.mean([s["chroma_dev"] for s in stats]))
            std_chroma = float(np.std([s["chroma_dev"] for s in stats]))

            summary["classes"][cname] = {
                "samples": len(stats),
                "crack_presence_rate": round(presence, 4),
                "mechanism_adherence_rate": round(adhere, 4),
                "mechanism_error_rate": round(err_rate, 4),
                "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
                "spatial_trajectory_recall": round(mean_recall, 4),
                "spatial_trajectory_recall_std": round(std_recall, 4),
                "trajectory_mae_px": round(traj_mae, 3),
                "trajectory_mae_std_px": round(traj_mae_std, 3),
                "trajectory_rmse_px": round(traj_rmse, 3),
                "trajectory_hausdorff95_px": round(traj_h95, 3),
                "angle_error_mae_deg": round(ang_mae, 2),
                "angle_error_std_deg": round(ang_std, 2),
                "angle_error_rmse_deg": round(ang_rmse, 2),
                "curvature_error_mae_deg": round(curv_mae, 2),
                "curvature_error_std_deg": round(curv_std, 2),
                "curvature_error_rmse_deg": round(curv_rmse, 2),
                "taper_error_mae": round(taper_mae, 3),
                "taper_error_std": round(taper_err_std, 3),
                "taper_error_rmse": round(taper_rmse, 3),
                "mean_max_length_px": round(mean_len, 1),
                "std_max_length_px": round(std_len, 1),
                "mean_continuity_ratio": round(mean_cr, 3),
                "mean_commanded_curvature_deg": round(cmd_curv, 1),
                "mean_curvature_deg": round(mean_curv, 1),
                "std_curvature_deg": round(std_curv, 1),
                "mean_commanded_taper_ratio": round(cmd_taper, 2),
                "mean_taper_ratio": round(mean_taper, 2),
                "std_taper_ratio": round(std_taper, 2),
                "spurious_background_fissure_rate": round(mean_fp_area, 5),
                "mean_chromatic_deviation": round(mean_chroma, 2),
                "std_chromatic_deviation": round(std_chroma, 2),
            }
            print(
                f"Class {cid} ({cname:20s}) [N={len(stats):3d}]: Adhere={adhere*100:5.1f}% (Err={err_rate*100:4.1f}%), "
                f"Recall={mean_recall*100:5.1f}%+/-{std_recall*100:3.1f}%, "
                f"TrajMAE={traj_mae:4.2f}+/-{traj_mae_std:4.2f}px (RMSE={traj_rmse:4.2f}px), "
                f"AngMAE={ang_mae:4.1f}deg, Taper={mean_taper:.2f} (Err={taper_mae:.2f})"
            )

    # Save 1,400-row per-sample CSV log
    if per_sample_rows:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"\n[SUCCESS] Saved {len(per_sample_rows)}-row per-sample error log to: {OUT_CSV}")

    # Store 7x7 confusion matrix and per-class precision/recall/F1
    summary["confusion_matrix"]["matrix_counts"] = conf_mat.tolist()
    summary["confusion_matrix"]["class_order"] = [CLASSES[i] for i in range(7)]
    per_class_clf = {}
    for i in range(7):
        tp = int(conf_mat[i, i])
        fn = int(conf_mat[i, :].sum() - tp)
        fp = int(conf_mat[:, i].sum() - tp)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        per_class_clf[CLASSES[i]] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "classification_error_rate": round(1.0 - rec, 4),
        }
    summary["confusion_matrix"]["per_class_classification_metrics"] = per_class_clf
    overall_acc = float(np.trace(conf_mat) / max(1, conf_mat.sum()))
    summary["confusion_matrix"]["overall_accuracy"] = round(overall_acc, 4)
    summary["confusion_matrix"]["overall_error_rate"] = round(1.0 - overall_acc, 4)
    print(f"Overall 7-Class Accuracy (N={int(conf_mat.sum())}): {overall_acc*100:.2f}% (Overall Error Rate = {(1.0-overall_acc)*100:.2f}%)")

    # Generate visual error plots
    generate_error_plots(summary, conf_mat, per_class_stats)

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.with_fidelity:
        print(f'[SUCCESS] Saved structural results to {OUT_JSON}; fidelity not requested.')
        return

    # Class-matched FID and KID across all 1,400 generated images vs 1,400 real images (200 per class)
    n_per_class = max(16, len(all_fake_paths) // 7)
    by_class = {c: [] for c in range(7)}
    with args.reference_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = int(r["class_id"])
            if len(by_class[cid]) < n_per_class:
                by_class[cid].append(ROOT / r["path"])
    real_paths = [p for cid in range(7) for p in by_class[cid]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nComputing Class-Matched FID and KID ({len(all_fake_paths)} generated vs {len(real_paths)} real images)...")
    # Standard backend shared with Experiment 5; identical preprocessing on both inputs.
    # Use isolated temporary folders so stale references cannot contaminate reruns.
    import tempfile
    import shutil
    import torch_fidelity
    try:
        with tempfile.TemporaryDirectory(dir=args.output_dir) as work:
            real_dir, fake_dir = Path(work) / 'real', Path(work) / 'fake'
            real_dir.mkdir()
            fake_dir.mkdir()
            for folder, paths in ((real_dir, real_paths), (fake_dir, all_fake_paths)):
                for idx, path in enumerate(paths):
                    shutil.copyfile(path, folder / f'{idx:06d}.png')
            raw = torch_fidelity.calculate_metrics(input1=str(fake_dir), input2=str(real_dir),
                cuda=torch.cuda.is_available(), fid=True, kid=True, batch_size=8,
                kid_subsets=8, kid_subset_size=min(100, len(real_paths), len(all_fake_paths)),
                verbose=False)
        fid_kid = {k: float(v) for k, v in raw.items()}
        summary['protocol']['fidelity_status'] = 'completed'
        summary['protocol']['reference_csv'] = str(args.reference_csv.resolve())
        summary['protocol']['reference_independence'] = 'User-supplied; must verify original-image training exclusion'
    except Exception as exc:
        summary['protocol']['fidelity_status'] = 'failed'
        summary['protocol']['fidelity_error'] = str(exc)
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        raise
    summary["fidelity"] = fid_kid
    print(f"FID: {fid_kid['frechet_inception_distance']}")
    print(f"KID: {fid_kid['kernel_inception_distance_mean']} +/- {fid_kid['kernel_inception_distance_std']}")

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Saved full evaluation & error statistics to: {OUT_JSON}")


if __name__ == "__main__":
    main()
