"""
Unified Replication & On-Demand Image Generation Toolkit for All 6 Concrete Crack DDPM Experiments (Exp 1 to Exp 6).

Allows any reviewer or user to:
  1. Choose any experiment (`--experiment 1..6`)
  2. Choose any specific crack type (`--crack-type flexural`, `--crack-type torsion`, `--crack-type all`, etc.)
  3. Generate any number of images (`--count 1234`, `--count 16`, etc.) in memory-safe mini-batches
  4. Use any random seed (`--seed 2026` to replicate the exact report images, or any new seed for fresh images)

Examples:
  # Generate 1234 new flexural crack images from Experiment 5 (Level B)
  python replicate_and_generate_suite/generate_cracks.py --experiment 5 --crack-type flexural --count 1234 --seed 999

  # Generate 16 flexure_shear crack images from Experiment 6 (Level C: Spatial-Physics)
  python replicate_and_generate_suite/generate_cracks.py --experiment 6 --crack-type flexure_shear --count 16 --seed 2026

  # Replicate all 7 classes of Experiment 6 (16 images per class)
  python replicate_and_generate_suite/generate_cracks.py --experiment 6 --crack-type all --count 16 --seed 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

SUITE_DIR = Path(__file__).resolve().parent
ROOT = SUITE_DIR.parent
sys.path.insert(0, str(SUITE_DIR / "core"))
sys.path.insert(0, str(SUITE_DIR))
sys.path.insert(0, str(ROOT / "src"))

from diffusion import GaussianDiffusion
from model import SmallUNet
from model_attention import AttentionUNet256
from physics_trajectory_solver import solve_physical_trajectory
from exp4_level_a_morphology_256.sample_crack_type_256 import sample_ddim_cfg_batch
from exp6_level_c_spatial_physics_256.sample_exp6_spatial_physics_256 import ddim_sample_dual_cfg

CHECKPOINTS = {
    1: ROOT / "models/weights/exp1_unconditional_64/exp1_unconditional_64_epoch050.pt",
    2: ROOT / "models/weights/exp2_conditional_64/exp2_conditional_64_epoch050.pt",
    3: ROOT / "models/weights/exp3_conditional_128/exp3_conditional_128_epoch050.pt",
    4: ROOT / "models/weights/exp4_level_a_morphology_256/exp4_level_a_morphology_256_epoch050.pt",
    5: ROOT / "models/weights/exp5_level_b_mechanisms_256/exp5_level_b_mechanisms_256_epoch025.pt",
    6: ROOT / "models/weights/exp6_level_c_spatial_physics_256/exp6_level_c_spatial_physics_256_epoch015.pt",
}

EXPERIMENT_CLASSES = {
    1: {0: "unconditional"},
    2: {0: "non_cracked", 1: "cracked"},
    3: {0: "non_cracked", 1: "cracked"},
    4: {
        0: "non_cracked",
        1: "longitudinal",
        2: "transverse",
        3: "diagonal",
        4: "alligator_map",
    },
    5: {
        0: "non_cracked",
        1: "flexural",
        2: "flexure_shear",
        3: "web_shear",
        4: "torsion",
        5: "corrosion_splitting",
        6: "shrinkage_map",
    },
    6: {
        0: "non_cracked",
        1: "flexural",
        2: "flexure_shear",
        3: "web_shear",
        4: "torsion",
        5: "corrosion_splitting",
        6: "shrinkage_map",
    },
}


@torch.no_grad()
def sample_small_unet_ddpm(
    model: SmallUNet,
    diffusion: GaussianDiffusion,
    count: int,
    class_id: int | None,
    image_size: int,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    all_imgs = []
    generated = 0
    pbar = tqdm(total=count, desc="Generating images")
    while generated < count:
        cur_n = min(batch_size, count - generated)
        images = torch.randn(cur_n, 3, image_size, image_size, device=device)
        labels = (
            torch.full((cur_n,), class_id, device=device, dtype=torch.long)
            if class_id is not None
            else None
        )
        for timestep in reversed(range(diffusion.timesteps)):
            t = torch.full((cur_n,), timestep, device=device, dtype=torch.long)
            pred_noise = model(images, t, labels) if labels is not None else model(images, t)
            beta = diffusion.betas[timestep]
            alpha = diffusion.alphas[timestep]
            alpha_bar = diffusion.alpha_bars[timestep]
            mean = (images - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            images = mean if timestep == 0 else mean + torch.sqrt(beta) * torch.randn_like(images)
        all_imgs.append(images.clamp(-1.0, 1.0).cpu())
        generated += cur_n
        pbar.update(cur_n)
    pbar.close()
    return torch.cat(all_imgs, dim=0)


def resolve_target_classes(exp_id: int, crack_type_arg: str) -> list[tuple[int, str]]:
    class_map = EXPERIMENT_CLASSES[exp_id]
    arg_clean = crack_type_arg.strip().lower()
    if arg_clean == "all":
        return list(class_map.items())

    # Allow numeric ID or string name (also allow "flexure" alias for "flexural")
    if arg_clean == "flexure":
        arg_clean = "flexural"

    if arg_clean.isdigit():
        cid = int(arg_clean)
        if cid in class_map:
            return [(cid, class_map[cid])]

    for cid, cname in class_map.items():
        if cname.lower() == arg_clean:
            return [(cid, cname)]

    valid = ", ".join(f"{k}:{v}" for k, v in class_map.items())
    raise SystemExit(
        f"[ERROR] Invalid --crack-type '{crack_type_arg}' for Experiment {exp_id}.\n"
        f"Valid choices for Experiment {exp_id} are: all, {valid}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Concrete Crack Image Generator for Experiments 1 through 6."
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        required=True,
        help="Experiment number (1 to 6)",
    )
    parser.add_argument(
        "--crack-type",
        type=str,
        default="all",
        help="Specific crack type to generate (e.g. 'flexural', 'flexure_shear', 'torsion', 'cracked', or 'all')",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of images to generate for the selected crack type(s) (e.g. 16, 100, 1234)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for repeatable runs with identical code, settings and hardware",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="DDIM sampling steps for Experiments 4, 5, 6 (default: 50)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=2.5,
        help="Classifier-Free Guidance scale for Experiments 4, 5, 6 (default: 2.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output folder (defaults to replicate_and_generate_suite/generated_outputs/exp<N>)",
    )
    args = parser.parse_args()
    if args.count < 1 or not 2 <= args.steps <= 1000 or not math.isfinite(args.cfg_scale) or args.cfg_scale < 0 or not 0 <= args.seed < 2**32:
        parser.error('Use positive count, 2..1000 steps, finite nonnegative guidance and seed 0..2^32-1')

    exp_id = args.experiment
    ckpt_path = CHECKPOINTS[exp_id]
    if not ckpt_path.exists():
        raise SystemExit(f"[ERROR] Checkpoint not found at: {ckpt_path}")

    target_classes = resolve_target_classes(exp_id, args.crack_type)

    out_base = (
        Path(args.output_dir)
        if args.output_dir
        else SUITE_DIR / "generated_outputs" / f"exp{exp_id}"
    )
    out_base.mkdir(parents=True, exist_ok=True)
    manifest_path=out_base/'inference_manifest.json'
    if any(out_base.iterdir()):
        parser.error('Output directory is not empty; choose a fresh --output-dir to preserve prior results')
    h=hashlib.sha256()
    with ckpt_path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    run_record=dict(experiment=exp_id,count_per_class=args.count,classes=[name for _,name in target_classes],
                    seed=args.seed,steps=args.steps if exp_id>=4 else 1000,cfg_scale=args.cfg_scale if exp_id>=4 else None,
                    checkpoint_path=str(ckpt_path.relative_to(ROOT)),checkpoint_sha256=h.hexdigest(),
                    torch_version=torch.__version__,status='running',
                    scope='Historical pretrained inference; no retraining or independent mechanism-validation claim')
    manifest_path.write_text(json.dumps(run_record,indent=2),encoding='utf-8')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================================")
    print(f" Experiment {exp_id} Pretrained Generator (NO RETRAINING REQUIRED)")
    print(f" Device     : {device}")
    print(f" Checkpoint : {ckpt_path.name}")
    print(f" Crack Types: {[name for _, name in target_classes]}")
    print(f" Count/Class: {args.count} images | Seed: {args.seed}")
    print(f" Output Dir : {out_base}")
    print(f"==================================================================")

    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if exp_id in (1, 2, 3):
        image_size = 64 if exp_id in (1, 2) else 128
        num_classes = None if exp_id == 1 else 2
        model = SmallUNet(num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        for cid, cname in target_classes:
            torch.manual_seed(args.seed + cid)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + cid)
            class_arg = None if exp_id == 1 else cid
            imgs = sample_small_unet_ddpm(
                model, diffusion, args.count, class_arg, image_size, device, batch_size=16
            )
            class_dir = out_base / cname
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(imgs.shape[0]):
                save_image(
                    imgs[idx],
                    class_dir / f"{cname}_seed{args.seed}_{idx:04d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            grid_n = min(16, imgs.shape[0])
            save_image(
                imgs[:grid_n],
                out_base / f"{cname}_preview_grid.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            print(f"[SUCCESS] Saved {imgs.shape[0]} images to: {class_dir}")

    elif exp_id in (4, 5):
        num_classes = 5 if exp_id == 4 else 7
        model = AttentionUNet256(base_channels=40, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        for cid, cname in target_classes:
            print(f"\n[*] Generating {args.count} images for Class {cid} ({cname})...")
            class_dir = out_base / cname
            class_dir.mkdir(parents=True, exist_ok=True)

            # Generate and save incrementally in chunks so even --count 1234 uses < 3.5 GB RAM/VRAM
            chunk_size = 8
            first_grid_imgs = []
            generated = 0
            pbar = tqdm(total=args.count, desc=f"Sampling {cname}")
            while generated < args.count:
                cur_n = min(chunk_size, args.count - generated)
                imgs = sample_ddim_cfg_batch(
                    model,
                    diffusion,
                    cur_n,
                    cid,
                    device,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    sub_batch_size=4,
                )
                for i in range(cur_n):
                    img_idx = generated + i
                    save_image(
                        imgs[i],
                        class_dir / f"sample_{img_idx:04d}.png",
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    if len(first_grid_imgs) < 16:
                        first_grid_imgs.append(imgs[i])
                generated += cur_n
                pbar.update(cur_n)
            pbar.close()

            if first_grid_imgs:
                grid_tensor = torch.stack(first_grid_imgs, dim=0)
                save_image(
                    grid_tensor,
                    out_base / f"{cname}_256_grid.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
            print(f"[SUCCESS] Saved {args.count} individual 256x256 PNGs to: {class_dir}")

    elif exp_id == 6:
        model = AttentionUNet256(
            in_channels=5, out_channels=3, base_channels=40, num_classes=7, cfg_dropout=0.0
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        for cid, cname in target_classes:
            print(f"\n[*] Generating {args.count} spatial-physics controlled images for Class {cid} ({cname})...")
            class_dir = out_base / cname
            phys_dir = out_base / f"{cname}_physics_masks"
            class_dir.mkdir(parents=True, exist_ok=True)
            phys_dir.mkdir(parents=True, exist_ok=True)

            chunk_size = 8
            first_grid_imgs = []
            generated = 0
            pbar = tqdm(total=args.count, desc=f"Sampling {cname} (Level C)")
            while generated < args.count:
                cur_n = min(chunk_size, args.count - generated)
                phys_list = []
                for i in range(cur_n):
                    img_idx = generated + i
                    phys_np, _ = solve_physical_trajectory(
                        cid, seed=args.seed + 1974 + cid * 100 + img_idx
                    )
                    phys_list.append(torch.from_numpy(phys_np))
                sub_phys = torch.stack(phys_list, dim=0)

                torch.manual_seed(args.seed + cid * 10 + generated)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + cid * 10 + generated)

                sub_samples = ddim_sample_dual_cfg(
                    model,
                    diffusion,
                    cid,
                    sub_phys,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    device=device,
                ).cpu()

                for i in range(cur_n):
                    img_idx = generated + i
                    save_image(sub_samples[i], class_dir / f"sample_{img_idx:04d}.png")
                    np.save(phys_dir / f"phys_{img_idx:04d}.npy", sub_phys[i].numpy())
                    if len(first_grid_imgs) < 16:
                        first_grid_imgs.append(sub_samples[i])

                generated += cur_n
                pbar.update(cur_n)
            pbar.close()

            if first_grid_imgs:
                grid = make_grid(torch.stack(first_grid_imgs, dim=0), nrow=4, padding=4)
                save_image(grid, out_base / f"{cname}_256_grid.png")
            print(f"[SUCCESS] Saved {args.count} individual 256x256 PNGs + physics masks to: {class_dir}")


    run_record['status']='completed'
    manifest_path.write_text(json.dumps(run_record,indent=2),encoding='utf-8')


if __name__ == "__main__":
    main()
