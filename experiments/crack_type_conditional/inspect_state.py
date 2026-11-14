"""Read-only inspection of concrete_ddpm state. Writes nothing into the project."""
import csv, sys, time, hashlib
from collections import Counter
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(r"D:\Downloads\concrete_ddpm")
sys.path.insert(0, str(ROOT / "src"))
from model import SmallUNet
from diffusion import GaussianDiffusion

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import torchmetrics; print("torchmetrics", torchmetrics.__version__)

E = ROOT / "experiments"
ckpts = {
    "exp1": (E / "unconditional_ddpm/checkpoints/ddpm_epoch_050.pt", None),
    "exp2": (E / "conditional_ddpm/checkpoints/conditional_epoch_050.pt", 2),
    "exp3": (E / "conditional_ddpm_128_augmented/checkpoints/conditional_128_augmented_epoch_050.pt", 2),
}
for name, (p, nc) in ckpts.items():
    c = torch.load(p, map_location="cpu")
    m = SmallUNet(num_classes=nc)
    sd = c["model"] if "model" in c else c
    r = m.load_state_dict(sd, strict=True)
    finite = all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
    print(name, "keys", list(c.keys()), "epoch", c.get("epoch"), "loss", c.get("loss"),
          "params", sum(x.numel() for x in m.parameters()), "finite", finite)

# Validation composition: what evaluate_metrics actually uses (first 1000 sorted PNGs)
val = {Path(r["path"]).name: r for r in csv.DictReader(open(ROOT / "data/processed/validation.csv"))}
files = sorted((ROOT / "data/processed/validation").glob("*.png"))[:1000]
print("val total", len(val), Counter(r["label"] for r in val.values()))
print("first1000 label", Counter(val[f.name]["label"] for f in files),
      "surface", Counter(val[f.name]["surface"] for f in files))
# leakage: validation originals vs exp3 train originals
tr = list(csv.DictReader(open(ROOT / "data/processed_128/train.csv")))
print("exp3 train", len(tr), Counter((r["source"], r["label"]) for r in tr))
print("val/train original overlap", len({r["original_path"] for r in val.values()} & {r["original_path"] for r in tr}))
print("val size", Image.open(files[0]).size)

# 32 partial images
pdir = E / "conditional_ddpm_128_augmented/outputs/evaluation/conditional_128_augmented_cracked"
pf = sorted(pdir.glob("*.png"))
print("partial", len(pf), pf[0].name, pf[-1].name, {Image.open(f).size for f in pf},
      "dup hashes", len(pf) - len({hashlib.md5(f.read_bytes()).hexdigest() for f in pf}))

# Determinism check: regenerate seed 2026 batch of 32 exactly as generate_eval_128.py, to scratch
from generate_eval_128 import sample
dev = torch.device("cuda")
m = SmallUNet(num_classes=2).to(dev); m.load_state_dict(torch.load(ckpts["exp3"][0], map_location=dev)["model"]); m.eval()
d = GaussianDiffusion(device=dev)
torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
t0 = time.time(); imgs = sample(m, d, 32, 1, dev); torch.cuda.synchronize(); dt = time.time() - t0
print(f"32 images x 1000 steps: {dt:.1f}s, peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
out = Path(r"D:\ANTIG\concrete_ddpm_scratch\determinism"); out.mkdir(parents=True, exist_ok=True)
from torchvision.utils import save_image
diffs = []
for i, im in enumerate(imgs):
    save_image(im, out / f"image_{i:04d}.png", normalize=True, value_range=(-1, 1))
    a = np.asarray(Image.open(out / f"image_{i:04d}.png"), dtype=np.int16)
    b = np.asarray(Image.open(pf[i]), dtype=np.int16)
    diffs.append(np.abs(a - b).mean())
print("mean abs pixel diff vs existing per image (first 8):", [round(x, 2) for x in diffs[:8]], "max", max(diffs))
