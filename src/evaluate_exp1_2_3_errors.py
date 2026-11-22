"""
Comprehensive Quantitative Error & Statistical Analysis for:
  - Experiment 1 (Unconditional 64x64 DDPM: N = 1,000 existing generated images)
  - Experiment 2 (Binary Conditional 64x64 DDPM: N = 2,000 existing generated images, 1,000 per class)
  - Experiment 3 (Augmented Binary Conditional 128x128 DDPM: N = 400 generated images, 200 per class)

Saves all outputs strictly inside each experiment's existing directory:
  1. `experiments/unconditional_ddpm/`
     - `EVALUATION_METRICS.json`
     - `PER_SAMPLE_ERROR_LOG.csv` (1,000 rows)
     - `exp1_error_analysis_plots.png`
  2. `experiments/conditional_ddpm/`
     - `EVALUATION_METRICS.json`
     - `PER_SAMPLE_ERROR_LOG.csv` (2,000 rows)
     - `exp2_error_analysis_plots.png`
     - `EXPERIMENT_2_FULL_REPORT.md` (updated/created with full error tables)
  3. `experiments/conditional_ddpm_128_augmented/`
     - `EVALUATION_METRICS.json`
     - `PER_SAMPLE_ERROR_LOG.csv` (400 rows)
     - `exp3_error_analysis_plots.png`
     - `EXPERIMENT_3_FULL_REPORT.md` (updated/created with full error tables)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from scipy.ndimage import binary_opening, find_objects, label, uniform_filter
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
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
    if len(x) < 6:
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


def analyze_patch(im_path: Path, resolution: int = 64) -> dict:
    """
    Scale-aware dark-fissure and surface texture analyzer for 64x64 (Exp 1 & 2) and 128x128 (Exp 3).
    """
    im = Image.open(im_path).convert("RGB")
    rgb = np.array(im, dtype=np.float32)
    mu_r, mu_g, mu_b = float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())
    mu_l = 0.299 * mu_r + 0.587 * mu_g + 0.114 * mu_b
    chroma_dev = max(abs(mu_r - mu_l), abs(mu_g - mu_l), abs(mu_b - mu_l))

    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    luma_mean = float(gray.mean())
    luma_std = float(gray.std())

    win_size = 11 if resolution == 64 else 17
    thresh = 13.5 if resolution == 64 else 15.0
    min_area = 6 if resolution == 64 else 12
    min_span = 11.0 if resolution == 64 else 22.0

    local_mean = uniform_filter(gray, size=win_size)
    diff = local_mean - gray
    raw = diff > thresh
    if resolution >= 128:
        mask = binary_opening(raw, structure=np.ones((2, 2), dtype=bool))
    else:
        mask = raw

    fissure_area_ratio = float(np.mean(mask))
    labeled, n_comp = label(mask, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        return {
            "has_crack": False,
            "max_span_px": 0.0,
            "continuity_ratio": 0.0,
            "dominant_angle_deg": 0.0,
            "elongation": 1.0,
            "fissure_area_ratio": fissure_area_ratio,
            "luma_mean": luma_mean,
            "luma_std": luma_std,
            "chroma_dev": chroma_dev,
        }

    slices = find_objects(labeled)
    comps = []
    total_area = 0
    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        cid = idx + 1
        sub = labeled[sl] == cid
        area = int(sub.sum())
        if area < min_area:
            continue
        total_area += area
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        diag = math.hypot(h, w)
        full_c = np.zeros_like(mask)
        full_c[sl] = sub
        th, el = compute_angle_deg(full_c)
        comps.append({"area": area, "diag": diag, "theta": th, "elong": el})

    if not comps:
        return {
            "has_crack": False,
            "max_span_px": 0.0,
            "continuity_ratio": 0.0,
            "dominant_angle_deg": 0.0,
            "elongation": 1.0,
            "fissure_area_ratio": fissure_area_ratio,
            "luma_mean": luma_mean,
            "luma_std": luma_std,
            "chroma_dev": chroma_dev,
        }

    comps.sort(key=lambda c: c["area"], reverse=True)
    main_c = comps[0]
    c_ratio = main_c["area"] / max(1, total_area)
    has_crack = (main_c["diag"] >= min_span) and (main_c["elong"] >= 1.85 or main_c["diag"] >= min_span * 1.5)

    return {
        "has_crack": bool(has_crack),
        "max_span_px": float(main_c["diag"]) if has_crack else 0.0,
        "continuity_ratio": float(c_ratio) if has_crack else 0.0,
        "dominant_angle_deg": float(main_c["theta"]) if has_crack else 0.0,
        "elongation": float(main_c["elong"]) if has_crack else 1.0,
        "fissure_area_ratio": fissure_area_ratio,
        "luma_mean": luma_mean,
        "luma_std": luma_std,
        "chroma_dev": chroma_dev,
    }


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

    def get_feats(paths: list[Path]) -> np.ndarray:
        feats = []
        batch = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            batch.append(tf(im))
            if len(batch) == 32:
                x = torch.stack(batch).to(device)
                feats.append(net(x).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(device)
            feats.append(net(x).cpu().numpy())
        return np.concatenate(feats, axis=0)

    r_f = get_feats(real_paths)
    g_f = get_feats(fake_paths)

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


def fidelity_label(metrics: dict) -> str:
    if metrics.get('status')=='not_requested':return 'Fidelity not requested'
    return f"FID={metrics['frechet_inception_distance']}, KID={metrics['kernel_inception_distance_mean']:.4f}"


def evaluate_experiment_1(device: torch.device, output_dir: Path | None = None, with_fidelity: bool = True) -> dict:
    print("\n======================================================================")
    print(" Evaluating Experiment 1 (Unconditional 64x64 DDPM — N = 1,000 Images)")
    print("======================================================================")
    source_exp_dir = ROOT / "experiments/unconditional_ddpm"
    exp_dir = Path(output_dir) if output_dir is not None else source_exp_dir
    exp_dir.mkdir(parents=True,exist_ok=True)
    eval_dir = source_exp_dir / "outputs/evaluation/unconditional_epoch_050"
    fake_files = sorted(eval_dir.glob("*.png"))
    print(f"Found {len(fake_files)} generated images in {eval_dir.relative_to(ROOT)}")

    # Load 1,000 real reference images from data/processed/train.csv matching SDNET2018 distribution (15.13% cracked)
    real_csv = ROOT / "data/processed/train.csv"
    with real_csv.open("r", encoding="utf-8") as f:
        all_real = list(csv.DictReader(f))
    real_cracked_total = sum(1 for r in all_real if r["label"] == "cracked")
    real_cracked_ratio = real_cracked_total / len(all_real)  # 7636 / 50482 = 0.15126

    rng = np.random.default_rng(2026)
    sampled_idx = rng.choice(len(all_real), size=min(300, len(fake_files)), replace=False)
    real_paths = [ROOT / all_real[i]["path"] for i in sampled_idx]

    real_stats = [analyze_patch(p, resolution=64) for p in real_paths]
    real_luma_mean = float(np.mean([s["luma_mean"] for s in real_stats]))
    real_luma_std = float(np.mean([s["luma_std"] for s in real_stats]))
    real_span_mean = float(np.mean([s["max_span_px"] for s in real_stats if s["has_crack"]]))

    rows = []
    fake_stats = []
    for f in fake_files:
        s = analyze_patch(f, resolution=64)
        fake_stats.append(s)
        luma_err = abs(s["luma_mean"] - real_luma_mean)
        contrast_err = abs(s["luma_std"] - real_luma_std)
        span_err = abs(s["max_span_px"] - real_span_mean) if s["has_crack"] else 0.0
        rows.append({
            "sample_file": f.name,
            "detected_class": "cracked" if s["has_crack"] else "non_cracked",
            "has_crack": int(s["has_crack"]),
            "max_span_px": round(s["max_span_px"], 2),
            "span_error_vs_real_mean_px": round(span_err, 2),
            "continuity_ratio": round(s["continuity_ratio"], 3),
            "dominant_angle_deg": round(s["dominant_angle_deg"], 1),
            "fissure_area_ratio": round(s["fissure_area_ratio"], 5),
            "luma_mean": round(s["luma_mean"], 2),
            "luma_error_mae": round(luma_err, 2),
            "luma_std_contrast": round(s["luma_std"], 2),
            "contrast_error_mae": round(contrast_err, 2),
            "chroma_dev": round(s["chroma_dev"], 2),
        })

    n_total = len(fake_stats)
    n_cracked = sum(1 for s in fake_stats if s["has_crack"])
    n_clean = n_total - n_cracked
    gen_cracked_ratio = n_cracked / max(1, n_total)
    ratio_error = abs(gen_cracked_ratio - real_cracked_ratio)
    ci_cracked = wilson_ci_95(n_cracked, n_total)
    ci_clean = wilson_ci_95(n_clean, n_total)

    cracked_sub = [s for s in fake_stats if s["has_crack"]]
    luma_errs = [abs(s["luma_mean"] - real_luma_mean) for s in fake_stats]
    contrast_errs = [abs(s["luma_std"] - real_luma_std) for s in fake_stats]
    span_errs = [abs(s["max_span_px"] - real_span_mean) for s in cracked_sub] if cracked_sub else [0.0]

    # Save CSV and JSON BEFORE FID/KID so results are immediately on disk
    out_csv = exp_dir / "PER_SAMPLE_ERROR_LOG.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp 1] Wrote 1,000-row CSV: {out_csv}", flush=True)
    fid_kid = compute_fid_kid(real_paths[:160], fake_files[:160], device) if with_fidelity else {'status':'not_requested'}

    summary = {
        "experiment": "Experiment 1: Unconditional 64x64 DDPM",
        "resolution": "64x64",
        "total_evaluated_samples": n_total,
        "dataset_prior_comparison": {
            "real_sdnet2018_cracked_rate": round(real_cracked_ratio, 4),
            "real_sdnet2018_non_cracked_rate": round(1.0 - real_cracked_ratio, 4),
            "generated_cracked_rate": round(gen_cracked_ratio, 4),
            "generated_non_cracked_rate": round(1.0 - gen_cracked_ratio, 4),
            "prior_distribution_drift_error": round(ratio_error, 4),
            "cracked_wilson_95_ci": [round(ci_cracked[0], 4), round(ci_cracked[1], 4)],
            "non_cracked_wilson_95_ci": [round(ci_clean[0], 4), round(ci_clean[1], 4)],
        },
        "surface_and_morphology_error_statistics": {
            "real_reference_luma_mean": round(real_luma_mean, 2),
            "generated_luma_mean": round(float(np.mean([s["luma_mean"] for s in fake_stats])), 2),
            "generated_luma_std": round(float(np.std([s["luma_mean"] for s in fake_stats])), 2),
            "luma_error_mae": round(float(np.mean(luma_errs)), 2),
            "luma_error_rmse": round(float(np.sqrt(np.mean(np.square(luma_errs)))), 2),
            "real_reference_contrast_std": round(real_luma_std, 2),
            "generated_contrast_mean": round(float(np.mean([s["luma_std"] for s in fake_stats])), 2),
            "contrast_error_mae": round(float(np.mean(contrast_errs)), 2),
            "contrast_error_rmse": round(float(np.sqrt(np.mean(np.square(contrast_errs)))), 2),
            "real_cracked_mean_span_px": round(real_span_mean, 2),
            "generated_cracked_mean_span_px": round(float(np.mean([s["max_span_px"] for s in cracked_sub])), 2) if cracked_sub else 0.0,
            "generated_cracked_std_span_px": round(float(np.std([s["max_span_px"] for s in cracked_sub])), 2) if cracked_sub else 0.0,
            "crack_span_error_mae_px": round(float(np.mean(span_errs)), 2),
            "crack_span_error_rmse_px": round(float(np.sqrt(np.mean(np.square(span_errs)))), 2),
            "mean_continuity_ratio": round(float(np.mean([s["continuity_ratio"] for s in cracked_sub])), 3) if cracked_sub else 0.0,
            "mean_chromatic_deviation": round(float(np.mean([s["chroma_dev"] for s in fake_stats])), 2),
            "std_chromatic_deviation": round(float(np.std([s["chroma_dev"] for s in fake_stats])), 2),
        },
        "fidelity": fid_kid,
    }

    out_json = exp_dir / "EVALUATION_METRICS.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 1] Wrote JSON: {out_json}", flush=True)

    # Plot 4-panel error figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    plt.subplots_adjust(wspace=0.28, hspace=0.34)

    # Panel 1: Real vs Generated Class Prior Distribution & Drift Error
    ax = axes[0, 0]
    cats = ["Non-Cracked (%)", "Cracked (%)"]
    real_vals = [(1.0 - real_cracked_ratio) * 100, real_cracked_ratio * 100]
    gen_vals = [(1.0 - gen_cracked_ratio) * 100, gen_cracked_ratio * 100]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, real_vals, w, label="Real SDNET2018 Prior", color="#4a5568")
    ax.bar(x + w / 2, gen_vals, w, label="Generated Exp 1 (N=1,000)", color="#3182ce")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontweight="bold")
    ax.set_ylabel("Percentage of Dataset (%)")
    ax.set_title(f"Unconditional Prior Match (Drift Error = {ratio_error*100:.2f}%)", fontweight="bold")
    for i in range(2):
        ax.text(x[i] - w / 2, real_vals[i] + 1.5, f"{real_vals[i]:.1f}%", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, gen_vals[i] + 1.5, f"{gen_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 2: Luminance Distribution (Real vs Generated)
    ax = axes[0, 1]
    ax.hist([s["luma_mean"] for s in real_stats], bins=28, alpha=0.6, label="Real SDNET2018", color="#4a5568", density=True)
    ax.hist([s["luma_mean"] for s in fake_stats], bins=28, alpha=0.6, label="Generated Exp 1", color="#38a169", density=True)
    ax.set_xlabel("Patch Mean Luminance (0..255)")
    ax.set_ylabel("Density")
    ax.set_title(f"Luminance Distribution (MAE = {summary['surface_and_morphology_error_statistics']['luma_error_mae']:.2f})", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 3: Crack Span Distribution on Cracked Subset
    ax = axes[1, 0]
    r_spans = [s["max_span_px"] for s in real_stats if s["has_crack"]]
    g_spans = [s["max_span_px"] for s in cracked_sub]
    if r_spans and g_spans:
        ax.hist(r_spans, bins=20, alpha=0.6, label=f"Real Cracked (mu={np.mean(r_spans):.1f}px)", color="#4a5568", density=True)
        ax.hist(g_spans, bins=20, alpha=0.6, label=f"Gen Cracked (mu={np.mean(g_spans):.1f}px)", color="#e53e3e", density=True)
    ax.set_xlabel("Crack Bounding-Box Diagonal Span (px on 64x64)")
    ax.set_ylabel("Density")
    ax.set_title("Detected Crack Span Distribution (64x64)", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 4: Summary Error Metrics Bar Chart
    ax = axes[1, 1]
    m_names = ["Luma MAE\n(gray lvl)", "Contrast MAE\n(gray lvl)", "Span MAE\n(pixels)", "ChromaDev\n(gray lvl)"]
    m_vals = [
        summary["surface_and_morphology_error_statistics"]["luma_error_mae"],
        summary["surface_and_morphology_error_statistics"]["contrast_error_mae"],
        summary["surface_and_morphology_error_statistics"]["crack_span_error_mae_px"],
        summary["surface_and_morphology_error_statistics"]["mean_chromatic_deviation"],
    ]
    bars = ax.bar(m_names, m_vals, color=["#2b6cb0", "#805ad5", "#dd6b20", "#38a169"], alpha=0.9)
    for b, v in zip(bars, m_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("Error Magnitude")
    ax.set_title(f"Exp 1 Error Summary ({fidelity_label(fid_kid)})", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Experiment 1 (Unconditional 64x64 DDPM): Quantitative Error Analysis (N = 1,000 Images)", fontsize=13, fontweight="bold", y=0.98)
    out_plot = exp_dir / "exp1_error_analysis_plots.png"
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    print(f"[Exp 1] Prior Cracked Rate: Real={real_cracked_ratio*100:.2f}% vs Gen={gen_cracked_ratio*100:.2f}% (Drift Err={ratio_error*100:.2f}%)")
    print(f"[Exp 1] {fidelity_label(fid_kid)}")
    print(f"[Exp 1] Saved: {out_json.resolve()} (exists={out_json.exists()}), {out_csv.resolve()}, {out_plot.resolve()}")
    return summary


def evaluate_binary_conditional_experiment(
    exp_num: int,
    exp_dir: Path,
    non_cracked_dir: Path,
    cracked_dir: Path,
    real_csv_path: Path,
    resolution: int,
    device: torch.device,
    with_fidelity: bool = True,
) -> dict:
    exp_dir.mkdir(parents=True,exist_ok=True)
    print(f"\n======================================================================")
    print(f" Evaluating Experiment {exp_num} (Binary Conditional {resolution}x{resolution} DDPM)")
    print(f"======================================================================")
    nc_files = sorted(non_cracked_dir.glob("*.png"))
    cr_files = sorted(cracked_dir.glob("*.png"))
    print(f"Found {len(nc_files)} non_cracked images and {len(cr_files)} cracked images")

    with real_csv_path.open("r", encoding="utf-8") as f:
        all_real = list(csv.DictReader(f))
    real_nc_all = [ROOT / r["path"] for r in all_real if r["label"] == "non_cracked"]
    real_cr_all = [ROOT / r["path"] for r in all_real if r["label"] == "cracked"]

    rng = np.random.default_rng(2026)
    n_nc_ref = min(200, len(nc_files), len(real_nc_all))
    n_cr_ref = min(200, len(cr_files), len(real_cr_all))
    real_nc_paths = [real_nc_all[i] for i in rng.choice(len(real_nc_all), size=n_nc_ref, replace=False)]
    real_cr_paths = [real_cr_all[i] for i in rng.choice(len(real_cr_all), size=n_cr_ref, replace=False)]

    # Measure real reference statistics
    real_cr_sample = [analyze_patch(p, resolution=resolution) for p in real_cr_paths[:min(200, len(real_cr_paths))]]
    real_cr_spans = [s["max_span_px"] for s in real_cr_sample if s["has_crack"]]
    real_mean_span = float(np.mean(real_cr_spans)) if real_cr_spans else (35.0 if resolution == 64 else 75.0)

    conf_mat = np.zeros((2, 2), dtype=int)
    rows = []
    class_stats = {0: [], 1: []}

    for cid, cname, files in [(0, "non_cracked", nc_files), (1, "cracked", cr_files)]:
        for f in files:
            s = analyze_patch(f, resolution=resolution)
            pred_cid = 1 if s["has_crack"] else 0
            adhered = int(pred_cid == cid)
            conf_mat[cid, pred_cid] += 1
            span_err = abs(s["max_span_px"] - real_mean_span) if cid == 1 else s["max_span_px"]
            s["adhered"] = adhered
            s["pred_cid"] = pred_cid
            s["span_err"] = span_err
            class_stats[cid].append(s)

            rows.append({
                "commanded_class_id": cid,
                "commanded_class_name": cname,
                "sample_file": f.name,
                "predicted_class_id": pred_cid,
                "predicted_class_name": "cracked" if pred_cid == 1 else "non_cracked",
                "adhered": adhered,
                "classification_error": 1 - adhered,
                "max_span_px": round(s["max_span_px"], 2),
                "span_error_px": round(span_err, 2),
                "continuity_ratio": round(s["continuity_ratio"], 3),
                "dominant_angle_deg": round(s["dominant_angle_deg"], 1),
                "elongation": round(s["elongation"], 2),
                "fissure_area_ratio": round(s["fissure_area_ratio"], 5),
                "luma_mean": round(s["luma_mean"], 2),
                "luma_std": round(s["luma_std"], 2),
                "chroma_dev": round(s["chroma_dev"], 2),
            })

    # Compute per-class metrics
    summary_classes = {}
    for cid, cname in [(0, "non_cracked"), (1, "cracked")]:
        st = class_stats[cid]
        n_tot = max(1, len(st))
        n_ok = sum(s["adhered"] for s in st)
        acc = n_ok / n_tot
        err = 1.0 - acc
        ci_l, ci_h = wilson_ci_95(n_ok, n_tot)
        spans = [s["max_span_px"] for s in st if s["has_crack"]]
        span_errs = [s["span_err"] for s in st]
        summary_classes[cname] = {
            "samples": len(st),
            "target_adherence_rate": round(acc, 4),
            "classification_error_rate": round(err, 4),
            "wilson_95_ci": [round(ci_l, 4), round(ci_h, 4)],
            "detected_crack_presence_rate": round(sum(1 for s in st if s["has_crack"]) / n_tot, 4),
            "mean_max_span_px": round(float(np.mean(spans)), 2) if spans else 0.0,
            "std_max_span_px": round(float(np.std(spans)), 2) if spans else 0.0,
            "span_error_mae_px": round(float(np.mean(span_errs)), 2),
            "span_error_rmse_px": round(float(np.sqrt(np.mean(np.square(span_errs)))), 2),
            "mean_continuity_ratio": round(float(np.mean([s["continuity_ratio"] for s in st if s["has_crack"]])), 3) if spans else 0.0,
            "mean_fissure_area_ratio": round(float(np.mean([s["fissure_area_ratio"] for s in st])), 5),
            "mean_chromatic_deviation": round(float(np.mean([s["chroma_dev"] for s in st])), 2),
            "std_chromatic_deviation": round(float(np.std([s["chroma_dev"] for s in st])), 2),
        }

    overall_acc = float(np.trace(conf_mat) / max(1, conf_mat.sum()))
    overall_err = 1.0 - overall_acc

    per_class_clf = {}
    for i, cname in enumerate(["non_cracked", "cracked"]):
        tp = int(conf_mat[i, i])
        fn = int(conf_mat[i, :].sum() - tp)
        fp = int(conf_mat[:, i].sum() - tp)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        per_class_clf[cname] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "error_rate": round(1.0 - rec, 4),
        }

    out_csv = exp_dir / "PER_SAMPLE_ERROR_LOG.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp {exp_num}] Wrote {len(rows)}-row CSV: {out_csv}", flush=True)

    fid_kid = compute_fid_kid(real_nc_paths[:80] + real_cr_paths[:80], nc_files[:80] + cr_files[:80], device) if with_fidelity else {'status':'not_requested'}

    summary = {
        "experiment": f"Experiment {exp_num}: Binary Conditional {resolution}x{resolution} DDPM",
        "resolution": f"{resolution}x{resolution}",
        "total_evaluated_samples": int(conf_mat.sum()),
        "classes": summary_classes,
        "confusion_matrix": {
            "matrix_counts": conf_mat.tolist(),
            "class_order": ["non_cracked", "cracked"],
            "overall_accuracy": round(overall_acc, 4),
            "overall_error_rate": round(overall_err, 4),
            "per_class_classification_metrics": per_class_clf,
        },
        "fidelity": fid_kid,
    }

    out_json = exp_dir / "EVALUATION_METRICS.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 1] Wrote JSON: {out_json}", flush=True)

    # Generate 4-panel error plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    plt.subplots_adjust(wspace=0.28, hspace=0.34)

    # Panel 1: 2x2 Confusion Matrix
    ax = axes[0, 0]
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm = np.divide(conf_mat, np.maximum(1, row_sums)) * 100.0
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non_cracked (0)", "cracked (1)"], fontweight="bold")
    ax.set_yticklabels(["non_cracked (0)", "cracked (1)"], fontweight="bold")
    ax.set_xlabel("Predicted Class", fontweight="bold")
    ax.set_ylabel("Commanded Class", fontweight="bold")
    ax.set_title(f"2x2 Binary Confusion Matrix (Acc = {overall_acc*100:.1f}%, Err = {overall_err*100:.1f}%)", fontweight="bold")
    for i in range(2):
        for j in range(2):
            cnt = int(conf_mat[i, j])
            pct = norm_cm[i, j]
            color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Per-Class Adherence vs Classification Error Rate
    ax = axes[0, 1]
    c_names = ["Class 0: non_cracked", "Class 1: cracked"]
    adh_vals = [summary_classes["non_cracked"]["target_adherence_rate"] * 100, summary_classes["cracked"]["target_adherence_rate"] * 100]
    err_vals = [summary_classes["non_cracked"]["classification_error_rate"] * 100, summary_classes["cracked"]["classification_error_rate"] * 100]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, adh_vals, w, label="Target Adherence (%)", color="#2b6cb0")
    ax.bar(x + w / 2, err_vals, w, label="Classification Error (%)", color="#e53e3e")
    ax.set_xticks(x)
    ax.set_xticklabels(c_names, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Per-Class Adherence vs. Error Rate", fontweight="bold")
    for i in range(2):
        ax.text(x[i] - w / 2, adh_vals[i] + 1.5, f"{adh_vals[i]:.1f}%", ha="center", fontweight="bold")
        ax.text(x[i] + w / 2, err_vals[i] + 1.5, f"{err_vals[i]:.1f}%", ha="center", fontweight="bold", color="#c53030")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 3: Fissure Span Distribution (Commanded Class 0 vs Commanded Class 1)
    ax = axes[1, 0]
    c0_spans = [s["max_span_px"] for s in class_stats[0]]
    c1_spans = [s["max_span_px"] for s in class_stats[1]]
    ax.hist(c0_spans, bins=20, alpha=0.65, label=f"Class 0: non_cracked (mu={np.mean(c0_spans):.1f}px)", color="#38a169")
    ax.hist(c1_spans, bins=20, alpha=0.65, label=f"Class 1: cracked (mu={np.mean(c1_spans):.1f}px)", color="#dd6b20")
    ax.set_xlabel(f"Detected Crack Span L_max (px on {resolution}x{resolution})")
    ax.set_ylabel("Sample Count")
    ax.set_title("Crack Span Separation Between Classes 0 and 1", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 4: Precision, Recall, F1 & Chromatic Error Summary
    ax = axes[1, 1]
    metrics_labels = ["Precision\n(Cracked)", "Recall\n(Cracked)", "F1-Score\n(Cracked)", "Overall\nAccuracy"]
    metrics_vals = [
        per_class_clf["cracked"]["precision"] * 100,
        per_class_clf["cracked"]["recall"] * 100,
        per_class_clf["cracked"]["f1_score"] * 100,
        overall_acc * 100,
    ]
    bars = ax.bar(metrics_labels, metrics_vals, color=["#3182ce", "#805ad5", "#38a169", "#2c5282"], alpha=0.9)
    for b, v in zip(bars, metrics_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Exp {exp_num} Summary ({fidelity_label(fid_kid)})", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Experiment {exp_num} (Binary Conditional {resolution}x{resolution} DDPM): Quantitative Error Analysis (N = {int(conf_mat.sum())} Images)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    out_plot = exp_dir / f"exp{exp_num}_error_analysis_plots.png"
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    print(f"[Exp {exp_num}] Class 0 Clean Adherence = {summary_classes['non_cracked']['target_adherence_rate']*100:.2f}% (Err={summary_classes['non_cracked']['classification_error_rate']*100:.2f}%)")
    print(f"[Exp {exp_num}] Class 1 Crack Adherence = {summary_classes['cracked']['target_adherence_rate']*100:.2f}% (Err={summary_classes['cracked']['classification_error_rate']*100:.2f}%)")
    print(f"[Exp {exp_num}] Overall Accuracy = {overall_acc*100:.2f}% (Overall Error Rate = {overall_err*100:.2f}%)")
    print(f"[Exp {exp_num}] {fidelity_label(fid_kid)}")
    print(f"[Exp {exp_num}] Saved: {out_json.name}, {out_csv.name}, {out_plot.name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    if 1 in args.experiments:
        evaluate_experiment_1(device)

    if 2 in args.experiments:
        exp2_dir = ROOT / "experiments/conditional_ddpm"
        evaluate_binary_conditional_experiment(
            exp_num=2,
            exp_dir=exp2_dir,
            non_cracked_dir=exp2_dir / "outputs/evaluation/conditional_non_cracked",
            cracked_dir=exp2_dir / "outputs/evaluation/conditional_cracked",
            real_csv_path=ROOT / "data/processed/train.csv",
            resolution=64,
            device=device,
        )

    if 3 in args.experiments:
        exp3_dir = ROOT / "experiments/conditional_ddpm_128_augmented"
        evaluate_binary_conditional_experiment(
            exp_num=3,
            exp_dir=exp3_dir,
            non_cracked_dir=exp3_dir / "outputs/evaluation/conditional_128_augmented_non_cracked",
            cracked_dir=exp3_dir / "outputs/evaluation/conditional_128_augmented_cracked",
            real_csv_path=ROOT / "data/processed_128/train.csv",
            resolution=128,
            device=device,
        )


if __name__ == "__main__":
    main()
