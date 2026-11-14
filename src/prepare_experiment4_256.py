"""
Prepare Experiment 4 (Level A): Morphological Crack-Type Dataset at Native 256x256.
Class 0: non_cracked
Class 1: longitudinal (~ 0 deg)
Class 2: transverse (~ 90 deg)
Class 3: diagonal (~ 45 or 135 deg)
Class 4: alligator_map (network / multi-directional)

Applies multi-tiered continuity filter:
- Discards noise (< 20 px)
- Preserves micro-cracks (>= 50 px) and major cracks
- Preserves alligator network segments (>= 25 px)
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed_256" / "train"
OUT_CSV = ROOT / "data" / "processed_256" / "train.csv"

def compute_morphology(mask_arr: np.ndarray):
    """
    Given binary mask of crack (256x256), returns:
    (is_valid, class_id, class_name, dominant_angle, max_length, c_ratio)
    """
    total_pixels = np.sum(mask_arr > 0)
    if total_pixels < 25:
        return False, 0, "none", 0.0, 0, 0.0

    labeled, num_features = label(mask_arr > 0)
    if num_features == 0:
        return False, 0, "none", 0.0, 0, 0.0

    slices = find_objects(labeled)
    comp_sizes = [np.sum(labeled == i + 1) for i in range(num_features)]
    max_idx = np.argmax(comp_sizes)
    max_slice = slices[max_idx]
    
    dy = max_slice[0].stop - max_slice[0].start
    dx = max_slice[1].stop - max_slice[1].start
    diag_len = float(np.sqrt(dx**2 + dy**2))
    c_ratio = float(comp_sizes[max_idx] / (total_pixels + 1e-5))

    # Noise rejection
    if diag_len < 20.0:
        return False, 0, "noise", 0.0, diag_len, c_ratio

    # Alligator / Map cracking check: multiple significant components / branches
    sig_components = sum(1 for sz in comp_sizes if sz >= 25)
    if sig_components >= 4 and c_ratio < 0.50:
        return True, 4, "alligator_map", 0.0, diag_len, c_ratio

    # Linear continuous crack check (>= 50 px)
    if diag_len >= 50.0 and c_ratio >= 0.55:
        # Compute central moments on main component
        main_mask = (labeled == (max_idx + 1))
        y, x = np.where(main_mask)
        xc, yc = x.mean(), y.mean()
        u20 = np.mean((x - xc)**2)
        u02 = np.mean((y - yc)**2)
        u11 = np.mean((x - xc) * (y - yc))
        theta_rad = 0.5 * np.arctan2(2 * u11, u20 - u02)
        theta_deg = float(np.degrees(theta_rad))
        if theta_deg < 0:
            theta_deg += 180.0

        # Classify by orientation
        # Horizontal: 0° ± 22.5° or 180° ± 22.5°
        if theta_deg <= 22.5 or theta_deg >= 157.5:
            return True, 1, "longitudinal", theta_deg, diag_len, c_ratio
        # Vertical: 90° ± 22.5° (67.5° to 112.5°)
        elif 67.5 <= theta_deg <= 112.5:
            return True, 2, "transverse", theta_deg, diag_len, c_ratio
        # Diagonal: 45° ± 22.5° (22.5° to 67.5°) or 135° ± 22.5° (112.5° to 157.5°)
        else:
            return True, 3, "diagonal", theta_deg, diag_len, c_ratio

    return False, 0, "unclassified", 0.0, diag_len, c_ratio

def process_deepcrack():
    """Extract 256x256 crops from DeepCrack dataset."""
    records = []
    p_img = ROOT / "data/raw/DeepCrack-master/dataset/train_img"
    p_lab = ROOT / "data/raw/DeepCrack-master/dataset/train_lab"
    if not p_img.exists():
        return records

    img_files = sorted(p_img.glob("*.*"))
    for f in tqdm(img_files, desc="Processing DeepCrack"):
        lab_file = p_lab / (f.stem + ".png")
        if not lab_file.exists():
            continue
        try:
            im = Image.open(f).convert("RGB")
            mask = Image.open(lab_file).convert("L")
            w, h = im.size
            # 544x384 -> slide 256x256 window with stride 96
            for y0 in range(0, h - 256 + 1, 96):
                for x0 in range(0, w - 256 + 1, 96):
                    box = (x0, y0, x0 + 256, y0 + 256)
                    crop_mask = np.array(mask.crop(box))
                    is_valid, cid, cname, theta, length, crat = compute_morphology(crop_mask)
                    if is_valid:
                        crop_im = im.crop(box)
                        hname = hashlib.md5(f"deepcrack_{f.stem}_{x0}_{y0}".encode()).hexdigest()[:16] + ".png"
                        out_path = OUT_DIR / hname
                        crop_im.save(out_path, optimize=True)
                        records.append({
                            "path": out_path.relative_to(ROOT).as_posix(),
                            "source": "DeepCrack",
                            "class_id": cid,
                            "class_name": cname,
                            "dominant_angle": f"{theta:.1f}",
                            "max_length": f"{length:.1f}",
                            "continuity_ratio": f"{crat:.3f}"
                        })
        except Exception as e:
            continue
    return records

def process_sdnet_noncracked(limit=6000):
    """Add uncracked concrete crops directly from SDNET2018 at native 256x256."""
    records = []
    p_uncracked = [
        ROOT / "data/raw/SDNET2018/images/D/UD",
        ROOT / "data/raw/SDNET2018/images/P/UP",
        ROOT / "data/raw/SDNET2018/images/W/UW",
    ]
    all_files = []
    for d in p_uncracked:
        if d.exists():
            all_files.extend(list(d.glob("*.jpg")))

    # Subsample evenly
    step = max(1, len(all_files) // limit)
    selected = all_files[::step][:limit]
    
    for f in tqdm(selected, desc="Processing SDNET2018 Non-cracked"):
        try:
            im = Image.open(f).convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.Resampling.LANCZOS)
            hname = hashlib.md5(f"sdnet_uncracked_{f.name}".encode()).hexdigest()[:16] + ".png"
            out_path = OUT_DIR / hname
            im.save(out_path, optimize=True)
            records.append({
                "path": out_path.relative_to(ROOT).as_posix(),
                "source": "SDNET2018",
                "class_id": 0,
                "class_name": "non_cracked",
                "dominant_angle": "0.0",
                "max_length": "0.0",
                "continuity_ratio": "0.000"
            })
        except Exception:
            continue
    return records

def process_sdnet_cracked():
    """Extract continuous cracks from SDNET2018 cracked subsets at native 256x256."""
    records = []
    p_cracked = [
        ROOT / "data/raw/SDNET2018/images/W/CW",
        ROOT / "data/raw/SDNET2018/images/D/CD",
        ROOT / "data/raw/SDNET2018/images/P/CP",
    ]
    all_files = []
    for d in p_cracked:
        if d.exists():
            all_files.extend(list(d.glob("*.jpg")))

    for f in tqdm(all_files, desc="Analyzing SDNET2018 Cracked"):
        try:
            im = Image.open(f).convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.array(im.convert("L"), dtype=np.float32)
            
            # Simple adaptive dark fissure extraction
            # Crack pixels are darker than local mean
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(arr, size=15)
            diff = local_mean - arr
            crack_mask = (diff > 18).astype(np.uint8)

            is_valid, cid, cname, theta, length, crat = compute_morphology(crack_mask)
            if is_valid:
                hname = hashlib.md5(f"sdnet_cracked_{f.name}".encode()).hexdigest()[:16] + ".png"
                out_path = OUT_DIR / hname
                im.save(out_path, optimize=True)
                records.append({
                    "path": out_path.relative_to(ROOT).as_posix(),
                    "source": "SDNET2018",
                    "class_id": cid,
                    "class_name": cname,
                    "dominant_angle": f"{theta:.1f}",
                    "max_length": f"{length:.1f}",
                    "continuity_ratio": f"{crat:.3f}"
                })
        except Exception:
            continue
    return records

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    # 1. DeepCrack high-res crops
    deepcrack_records = process_deepcrack()
    all_records.extend(deepcrack_records)
    print(f"DeepCrack valid crops: {len(deepcrack_records)}")

    # 2. SDNET2018 non-cracked (Class 0)
    sdnet_nc = process_sdnet_noncracked(limit=6000)
    all_records.extend(sdnet_nc)
    print(f"SDNET non-cracked: {len(sdnet_nc)}")

    # 3. SDNET2018 continuous cracks
    sdnet_cr = process_sdnet_cracked()
    all_records.extend(sdnet_cr)
    print(f"SDNET verified continuous cracks: {len(sdnet_cr)}")

    # Write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["path", "source", "class_id", "class_name", "dominant_angle", "max_length", "continuity_ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    # Class summary
    from collections import Counter
    counts = Counter(r["class_name"] for r in all_records)
    print("\n=== Level A Dataset Composition ===")
    for cname, cnt in sorted(counts.items()):
        print(f"  {cname:15s}: {cnt:5d} images")
    print(f"Total dataset: {len(all_records)} images at native 256x256")
    print(f"Saved manifest to: {OUT_CSV}")

if __name__ == "__main__":
    main()
