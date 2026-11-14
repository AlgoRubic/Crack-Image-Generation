"""Generate individual 128x128 conditional DDPM images for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion import GaussianDiffusion
from model import SmallUNet


@torch.no_grad()
def sample(model, diffusion, count, class_id, device, image_size=128):
    images = torch.randn(count, 3, image_size, image_size, device=device)
    labels = torch.full((count,), class_id, device=device, dtype=torch.long)
    for timestep in reversed(range(diffusion.timesteps)):
        t = torch.full((count,), timestep, device=device, dtype=torch.long)
        predicted_noise = model(images, t, labels)
        beta = diffusion.betas[timestep]
        alpha = diffusion.alphas[timestep]
        alpha_bar = diffusion.alpha_bars[timestep]
        mean = (images - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        images = mean if timestep == 0 else mean + torch.sqrt(beta) * torch.randn_like(images)
    return images.clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-name", choices=["cracked", "non_cracked"], required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to experiments/conditional_ddpm_128_augmented/outputs/evaluation/conditional_128_augmented_<class>")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (root / "experiments" / "conditional_ddpm_128_augmented" / "outputs" / "evaluation"
                      / f"conditional_128_augmented_{args.class_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic resume: batch b always uses seed (args.seed + b) and writes
    # images b*batch_size .. b*batch_size+n-1. A batch is skipped only if all of
    # its files already exist, so existing valid samples are never overwritten.
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "class_name": args.class_name,
        "count": args.count,
        "batch_size": args.batch_size,
        "seed_rule": f"seed = {args.seed} + batch_index",
        "base_seed": args.seed,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("class_name", "batch_size", "base_seed"):
            if previous.get(key) != manifest[key]:
                raise SystemExit(f"Refusing to resume: {key} differs from existing manifest "
                                 f"({previous.get(key)} != {manifest[key]})")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_id = 1 if args.class_name == "cracked" else 0

    model = SmallUNet(num_classes=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device)

    generated = 0
    batch_number = 0
    while generated < args.count:
        batch_count = min(args.batch_size, args.count - generated)
        paths = [output_dir / f"image_{generated + i:04d}.png" for i in range(batch_count)]
        if all(p.exists() for p in paths):
            generated += batch_count
            batch_number += 1
            print(f"{args.class_name}: batch {batch_number - 1} exists, skipped ({generated}/{args.count})", flush=True)
            continue
        seed = args.seed + batch_number
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        images = sample(model, diffusion, batch_count, class_id, device)
        for image, path in zip(images, paths):
            if not path.exists():
                save_image(image, path, normalize=True, value_range=(-1, 1))
            generated += 1
        batch_number += 1
        print(f"{args.class_name}: generated {generated}/{args.count}", flush=True)

    print(f"Device: {device}")
    print(f"Saved individual images to: {output_dir}")


if __name__ == "__main__":
    main()
