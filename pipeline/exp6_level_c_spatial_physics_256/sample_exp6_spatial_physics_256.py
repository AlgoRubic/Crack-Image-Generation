"""
Sample Experiment 6 (Level C): 50-Step DDIM + Dual Classifier-Free Guidance (Spatial Physics + Class)
across all 7 Structural Failure Mechanisms at Native 256x256.
Supports large-scale generation (e.g., --count 200 per class = 1,400 images total) with incremental
mini-batch disk saving and resume-skipping of existing valid samples.
Saves individual 256x256 PNG images per class, their paired analytical physics maps,
and 4x4 contact sheets into `experiments/exp6_level_c_spatial_physics/results/` (and `generated_samples/`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion import GaussianDiffusion
from model_attention import AttentionUNet256
from physics_trajectory_solver import CLASS_NAMES, solve_physical_trajectory


@torch.no_grad()
def ddim_sample_dual_cfg(
    model: AttentionUNet256,
    diffusion: GaussianDiffusion,
    class_id: int,
    phys_batch_01: torch.Tensor,
    steps: int = 50,
    cfg_scale: float = 2.5,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    50-Step Deterministic DDIM Sampling (sigma=0) with Dual Classifier-Free Guidance (Assumption C5):
      eps_guided = eps(x_t, t, null, S_null) + w * (eps(x_t, t, c, S) - eps(x_t, t, null, S_null))
    """
    model.eval()
    n = phys_batch_01.shape[0]
    shape = (n, 3, 256, 256)
    img = torch.randn(shape, device=device)

    phys_cond = (phys_batch_01.to(device) * 2.0) - 1.0  # (N, 2, 256, 256) in [-1, 1]
    phys_null = torch.full_like(phys_cond, -1.0)

    times = torch.linspace(diffusion.timesteps - 1, 0, steps, dtype=torch.long, device=device)
    c_cond = torch.full((n,), class_id, dtype=torch.long, device=device)
    c_null = torch.full((n,), model.null_token_idx, dtype=torch.long, device=device)

    for i in range(len(times)):
        t = times[i]
        t_batch = torch.full((n,), t, dtype=torch.long, device=device)

        x_cond_in = torch.cat([img, phys_cond], dim=1)
        x_null_in = torch.cat([img, phys_null], dim=1)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            eps_cond = model(x_cond_in, t_batch, c_cond, force_dropout=False)
            eps_uncond = model(x_null_in, t_batch, c_null, force_dropout=True)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        alpha_bar_t = diffusion.alpha_bars[t]
        pred_x0 = (img - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if i < len(times) - 1:
            t_prev = times[i + 1]
            alpha_bar_prev = diffusion.alpha_bars[t_prev]
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            img = pred_x0

    out_01 = (img.clamp(-1.0, 1.0) + 1.0) * 0.5
    lum = 0.299 * out_01[:, 0:1] + 0.587 * out_01[:, 1:2] + 0.114 * out_01[:, 2:3]
    out_01 = lum + 0.15 * (out_01 - lum)
    return out_01.clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/exp6_level_c_spatial_physics/checkpoints/exp6_spatial_physics_256_epoch_015.pt",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if sample_XXX.png already exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    model = AttentionUNet256(
        in_channels=5,
        out_channels=3,
        base_channels=40,
        num_classes=7,
        cfg_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, device=device)

    exp_dir = ROOT / "experiments/exp6_level_c_spatial_physics"
    out_dirs = [exp_dir / "results", exp_dir / "generated_samples"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    all_meta = {}
    t_start_all = time.time()
    chunk_size = 8

    for cid, cname in CLASS_NAMES.items():
        t_c0 = time.time()
        print(f"\n[{cid+1}/7] Sampling {args.count} physics-controlled images for Class {cid} ({cname}) [CFG={args.cfg_scale}]...")
        class_meta = []

        # Ensure output directories exist
        for base_d in out_dirs:
            (base_d / cname).mkdir(parents=True, exist_ok=True)
            (base_d / f"{cname}_physics_masks").mkdir(parents=True, exist_ok=True)

        # Precompute all analytical physics trajectories & metadata deterministically
        phys_list = []
        for idx in range(args.count):
            phys_np, m = solve_physical_trajectory(cid, seed=4000 + cid * 1000 + idx if idx >= 16 else 4000 + cid * 100 + idx)
            m["sample_index"] = idx
            m["sample_file"] = f"sample_{idx:03d}.png"
            m["physics_file"] = f"phys_{idx:03d}.npy"
            phys_list.append(phys_np)
            class_meta.append(m)

        # Stream mini-batches of 8 to GPU and immediately save to disk
        for s in range(0, args.count, chunk_size):
            end_idx = min(s + chunk_size, args.count)
            batch_indices = list(range(s, end_idx))

            # Check if all files in this mini-batch already exist in both directories
            if not args.overwrite:
                all_exist = True
                for idx in batch_indices:
                    for base_d in out_dirs:
                        im_p = base_d / cname / f"sample_{idx:03d}.png"
                        ph_p = base_d / f"{cname}_physics_masks" / f"phys_{idx:03d}.npy"
                        if not (im_p.exists() and ph_p.exists()):
                            all_exist = False
                            break
                    if not all_exist:
                        break
                if all_exist:
                    print(f"  [{cname}] Batch {s:03d}..{end_idx-1:03d} already exists on disk, skipping...")
                    continue

            sub_phys_np = np.stack([phys_list[i] for i in batch_indices], axis=0)
            sub_phys = torch.from_numpy(sub_phys_np)
            torch.manual_seed(2026 + cid * 1000 + s if s >= 16 else 2026 + cid * 10 + s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(2026 + cid * 1000 + s if s >= 16 else 2026 + cid * 10 + s)

            sub_samples = ddim_sample_dual_cfg(
                model,
                diffusion,
                cid,
                sub_phys,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                device=device,
            ).cpu()

            for local_i, idx in enumerate(batch_indices):
                for base_d in out_dirs:
                    im_p = base_d / cname / f"sample_{idx:03d}.png"
                    ph_p = base_d / f"{cname}_physics_masks" / f"phys_{idx:03d}.npy"
                    save_image(sub_samples[local_i], im_p)
                    np.save(ph_p, phys_list[idx])

            elapsed = time.time() - t_c0
            print(f"  [{cname}] Generated {end_idx:3d}/{args.count} images ({elapsed:.1f}s elapsed)")

        # Build 4x4 contact sheet from first 16 images
        first_16 = []
        for idx in range(min(16, args.count)):
            im_p = out_dirs[0] / cname / f"sample_{idx:03d}.png"
            with Image.open(im_p) as pil_im:
                first_16.append(to_tensor(pil_im.convert("RGB")))
        if first_16:
            grid = make_grid(torch.stack(first_16, dim=0), nrow=4, padding=4)
            for base_d in out_dirs:
                grid_path = base_d / f"{cname}_256_grid.png"
                save_image(grid, grid_path)

        all_meta[cname] = class_meta
        print(f"  [DONE] Completed {args.count} images for Class {cid} ({cname}) in {time.time() - t_c0:.1f}s")

    for base_d in out_dirs:
        (base_d / "commanded_physics_metadata.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")

    total_dt = time.time() - t_start_all
    print(f"\n[SUCCESS] Completed all {args.count * 7} images (7 classes x {args.count}) in {total_dt/60.0:.2f} minutes!")


if __name__ == "__main__":
    main()
