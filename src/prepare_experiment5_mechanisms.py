"""
Prepare Experiment 5 (Level B): 7-Class Structural Failure Mechanism Dataset at Native 256x256.

Classes:
  0: non_cracked          - Intact concrete surface
  1: flexural             - Bottom-tension initiated vertical crack, tapering upward
  2: flexure_shear        - Vertical tension crack at bottom that curves/bends diagonally (~45 deg) upward
  3: web_shear            - Single dominant diagonal tension crack (~30-60 deg) in the web
  4: torsion              - Parallel echelon / multi-strand spiral diagonal cracks across the face
  5: corrosion_splitting  - Longitudinal horizontal splitting crack along reinforcement axis (~0 deg)
  6: shrinkage_map        - Multi-directional interconnected crazing / drying shrinkage network

Includes canonical structural orientation alignment so that flexural and flexure-shear cracks
always initiate from the bottom tension zone and propagate upward.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt, find_objects, label, uniform_filter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed_exp5_mechanisms" / "train"
OUT_CSV = ROOT / "data" / "processed_exp5_mechanisms" / "train.csv"
VERIFY_GRID = ROOT / "experiments" / "structural_mechanism_ddpm" / "dataset_verification_grid.png"

CLASS_NAMES = {
    0: "non_cracked",
    1: "flexural",
    2: "flexure_shear",
    3: "web_shear",
    4: "torsion",
    5: "corrosion_splitting",
    6: "shrinkage_map",
}


def compute_angle_deg(mask_bool: np.ndarray) -> tuple[float, float]:
    """Return (angle_deg in [0, 180), elongation_ratio) from central moments."""
    y, x = np.where(mask_bool)
    if len(x) < 8:
        return 0.0, 1.0
    xc, yc = x.mean(), y.mean()
    u20 = float(np.mean((x - xc) ** 2)) + 1e-6
    u02 = float(np.mean((y - yc) ** 2)) + 1e-6
    u11 = float(np.mean((x - xc) * (y - yc)))
    theta_rad = 0.5 * np.arctan2(2.0 * u11, u20 - u02)
    theta_deg = float(np.degrees(theta_rad))
    if theta_deg < 0:
        theta_deg += 180.0

    # Eigenvalues of covariance matrix for elongation ratio
    common = np.sqrt((u20 - u02) ** 2 + 4.0 * (u11 ** 2))
    lam1 = 0.5 * (u20 + u02 + common)
    lam2 = max(1e-5, 0.5 * (u20 + u02 - common))
    elongation = float(np.sqrt(lam1 / lam2))
    return theta_deg, elongation


def angular_diff(a: float, b: float) -> float:
    """Minimum difference between two orientation angles in [0, 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def is_vertical(theta: float, tol: float = 22.5) -> bool:
    return (90.0 - tol) <= theta <= (90.0 + tol)


def is_horizontal(theta: float, tol: float = 22.5) -> bool:
    return theta <= tol or theta >= (180.0 - tol)


def is_diagonal(theta: float, tol: float = 22.5) -> bool:
    return ((45.0 - tol) <= theta <= (45.0 + tol)) or ((135.0 - tol) <= theta <= (135.0 + tol))


def classify_structural_mechanism(mask_arr: np.ndarray) -> tuple[bool, int, str, bool, dict]:
    """
    Analyze a 256x256 binary mask using 5 structural mechanics descriptors.
    Returns:
      (is_valid, class_id, class_name, flip_vertically, meta_dict)
    """
    binary = (mask_arr > 0).astype(np.uint8)
    total_pixels = int(np.sum(binary))
    if total_pixels < 30:
        return False, 0, "none", False, {}

    labeled, num_features = label(binary)
    if num_features == 0:
        return False, 0, "none", False, {}

    slices = find_objects(labeled)
    comp_info = []
    for idx in range(num_features):
        sl = slices[idx]
        if sl is None:
            continue
        c_mask = (labeled == (idx + 1))
        sz = int(np.sum(c_mask))
        dy = sl[0].stop - sl[0].start
        dx = sl[1].stop - sl[1].start
        diag = float(np.sqrt(dx * dx + dy * dy))
        comp_info.append((sz, diag, idx + 1, sl, c_mask))

    if not comp_info:
        return False, 0, "none", False, {}

    comp_info.sort(key=lambda x: x[0], reverse=True)
    max_sz, max_diag, max_label, max_slice, main_mask = comp_info[0]
    c_ratio = float(max_sz / (total_pixels + 1e-5))

    if max_diag < 25.0:
        return False, 0, "noise", False, {}

    # Significant components (size >= 30 px and span >= 25 px)
    sig_comps = [c for c in comp_info if c[0] >= 30 and c[1] >= 25.0]
    theta_main, elong_main = compute_angle_deg(main_mask)

    # 1. Check TORSION (Parallel Echelon Diagonal Cracks)
    # >= 2 significant components with diagonal inclination running roughly parallel
    diag_comps = []
    for sz, diag, lbl, sl, cm in sig_comps:
        if diag >= 38.0:
            th, el = compute_angle_deg(cm)
            if el >= 2.2 and is_diagonal(th, tol=25.0):
                diag_comps.append((sz, diag, th))

    if len(diag_comps) >= 2:
        th0 = diag_comps[0][2]
        parallel_matches = sum(1 for _, _, th in diag_comps[1:] if angular_diff(th0, th) <= 24.0)
        if parallel_matches >= 1 and 0.25 <= c_ratio <= 0.85:
            return True, 4, "torsion", False, {
                "angle": theta_main,
                "length": max_diag,
                "c_ratio": c_ratio,
                "curvature": 0.0,
                "taper": 1.0,
            }

    # 2. Check SHRINKAGE / MAP CRACKING (Isotropic polygonal crazing network)
    if len(sig_comps) >= 4 and c_ratio < 0.48:
        return True, 6, "shrinkage_map", False, {
            "angle": theta_main,
            "length": max_diag,
            "c_ratio": c_ratio,
            "curvature": 0.0,
            "taper": 1.0,
        }

    # For single-dominant fracture mechanisms, require a coherent main crack
    if max_diag < 55.0 or c_ratio < 0.52:
        return False, 0, "fragmented", False, {}

    # Compute piecewise curvature (lower half vs upper half of main crack bounding box)
    y_start, y_stop = max_slice[0].start, max_slice[0].stop
    y_mid = (y_start + y_stop) // 2
    lower_mask = np.zeros_like(main_mask)
    upper_mask = np.zeros_like(main_mask)
    lower_mask[y_mid:y_stop, :] = main_mask[y_mid:y_stop, :]
    upper_mask[y_start:y_mid, :] = main_mask[y_start:y_mid, :]

    th_lower, el_lower = compute_angle_deg(lower_mask)
    th_upper, el_upper = compute_angle_deg(upper_mask)
    delta_theta = angular_diff(th_lower, th_upper)

    # Compute Euclidean Distance Transform (EDT) for aperture width taper profile
    edt = distance_transform_edt(main_mask)
    w_lower = float(np.percentile(edt[lower_mask], 80)) if np.any(lower_mask) else 1.0
    w_upper = float(np.percentile(edt[upper_mask], 80)) if np.any(upper_mask) else 1.0
    taper_ratio = w_lower / max(0.5, w_upper)

    # 3. Check FLEXURE-SHEAR (Vertical tension crack that bends into a diagonal shear crack)
    lower_is_vert = is_vertical(th_lower, tol=24.0)
    upper_is_diag = is_diagonal(th_upper, tol=24.0)
    upper_is_vert = is_vertical(th_upper, tol=24.0)
    lower_is_diag = is_diagonal(th_lower, tol=24.0)

    if delta_theta >= 16.0 and max_diag >= 65.0:
        if lower_is_vert and upper_is_diag:
            # Already in canonical orientation (vertical at bottom tension face, diagonal in upper web)
            return True, 2, "flexure_shear", False, {
                "angle": theta_main,
                "length": max_diag,
                "c_ratio": c_ratio,
                "curvature": delta_theta,
                "taper": taper_ratio,
            }
        elif upper_is_vert and lower_is_diag:
            # Upside-down flexure-shear: flip vertically so vertical tension root is at the bottom
            return True, 2, "flexure_shear", True, {
                "angle": theta_main,
                "length": max_diag,
                "c_ratio": c_ratio,
                "curvature": delta_theta,
                "taper": 1.0 / max(0.1, taper_ratio),
            }

    # 4. Check FLEXURAL (Vertical bending crack, canonically widest/rooted at bottom tension zone)
    if is_vertical(theta_main, tol=21.0) and delta_theta < 20.0:
        touches_bottom = (y_stop >= 235)
        touches_top = (y_start <= 20)
        # Canonically orient so tension initiation / wider mouth is at the bottom
        flip_vert = False
        if (touches_top and not touches_bottom) or (not touches_bottom and w_upper > w_lower * 1.15):
            flip_vert = True
            taper_ratio = 1.0 / max(0.1, taper_ratio)

        return True, 1, "flexural", flip_vert, {
            "angle": theta_main,
            "length": max_diag,
            "c_ratio": c_ratio,
            "curvature": delta_theta,
            "taper": taper_ratio,
        }

    # 5. Check CORROSION SPLITTING (Horizontal longitudinal crack along rebar axis)
    if is_horizontal(theta_main, tol=21.0) and delta_theta < 20.0 and max_diag >= 60.0:
        return True, 5, "corrosion_splitting", False, {
            "angle": theta_main,
            "length": max_diag,
            "c_ratio": c_ratio,
            "curvature": delta_theta,
            "taper": taper_ratio,
        }

    # 6. Check WEB-SHEAR (Inclined diagonal tension crack in web)
    if is_diagonal(theta_main, tol=24.0) and max_diag >= 60.0:
        return True, 3, "web_shear", False, {
            "angle": theta_main,
            "length": max_diag,
            "c_ratio": c_ratio,
            "curvature": delta_theta,
            "taper": taper_ratio,
        }

    return False, 0, "unclassified", False, {}


def save_sample(im: Image.Image, flip_vert: bool, tag: str, source: str, cid: int, cname: str, meta: dict) -> dict:
    if flip_vert:
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    hname = hashlib.md5(tag.encode("utf-8")).hexdigest()[:16] + ".png"
    out_path = OUT_DIR / hname
    if not out_path.exists():
        im.save(out_path, optimize=True)
    return {
        "path": out_path.relative_to(ROOT).as_posix(),
        "source": source,
        "class_id": cid,
        "class_name": cname,
        "dominant_angle": f"{meta.get('angle', 0.0):.1f}",
        "max_length": f"{meta.get('length', 0.0):.1f}",
        "continuity_ratio": f"{meta.get('c_ratio', 0.0):.3f}",
        "curvature_deg": f"{meta.get('curvature', 0.0):.1f}",
        "taper_ratio": f"{meta.get('taper', 1.0):.2f}",
    }


def process_deepcrack_all() -> list[dict]:
    records = []
    splits = [
        ("train_img", "train_lab"),
        ("test_img", "test_lab"),
    ]
    base = ROOT / "data/raw/DeepCrack-master/dataset"
    for img_sub, lab_sub in splits:
        p_img = base / img_sub
        p_lab = base / lab_sub
        if not p_img.exists():
            continue
        for f in tqdm(sorted(p_img.glob("*.*")), desc=f"DeepCrack {img_sub}"):
            lab_file = p_lab / (f.stem + ".png")
            if not lab_file.exists():
                continue
            try:
                im = Image.open(f).convert("RGB")
                mask = Image.open(lab_file).convert("L")
                w, h = im.size
                for y0 in range(0, h - 256 + 1, 64):
                    for x0 in range(0, w - 256 + 1, 64):
                        box = (x0, y0, x0 + 256, y0 + 256)
                        crop_mask = np.array(mask.crop(box))
                        ok, cid, cname, flip_v, meta = classify_structural_mechanism(crop_mask)
                        if ok:
                            crop_im = im.crop(box)
                            rec = save_sample(
                                crop_im, flip_v,
                                f"deepcrack5_{img_sub}_{f.stem}_{x0}_{y0}",
                                "DeepCrack", cid, cname, meta
                            )
                            records.append(rec)
            except Exception:
                continue
    return records


def process_crackforest() -> list[dict]:
    records = []
    gt_dir = ROOT / "data/raw/CrackForest-dataset-master/groundTruth"
    im_dir = ROOT / "data/raw/CrackForest-dataset-master/image"
    if not gt_dir.exists() or not im_dir.exists():
        return records

    for mat_p in tqdm(sorted(gt_dir.glob("*.mat")), desc="CrackForest"):
        img_p = im_dir / (mat_p.stem + ".jpg")
        if not img_p.exists():
            continue
        try:
            m = loadmat(str(mat_p))
            gt = m["groundTruth"][0, 0]
            bounds = np.array(gt["Boundaries"], dtype=np.uint8)
            segs = (np.array(gt["Segmentation"]) > 1).astype(np.uint8)
            mask_full = np.maximum(bounds, segs) * 255

            im = Image.open(img_p).convert("RGB")
            mask_im = Image.fromarray(mask_full)
            w, h = im.size
            for y0 in range(0, h - 256 + 1, 64):
                for x0 in range(0, w - 256 + 1, 64):
                    box = (x0, y0, x0 + 256, y0 + 256)
                    crop_mask = np.array(mask_im.crop(box))
                    ok, cid, cname, flip_v, meta = classify_structural_mechanism(crop_mask)
                    if ok:
                        crop_im = im.crop(box)
                        rec = save_sample(
                            crop_im, flip_v,
                            f"cfd5_{mat_p.stem}_{x0}_{y0}",
                            "CrackForest", cid, cname, meta
                        )
                        records.append(rec)
        except Exception:
            continue
    return records


def process_rc10519_boxed(limit: int = 5000) -> list[dict]:
    """
    Process Reinforced Concrete Structure Crack Detection Data (RC_10519),
    restricting fissure mask extraction strictly inside human-annotated crack bounding boxes.
    """
    records = []
    base = ROOT / "data/raw/external_cracks/Reinforced concrete structure crack detection data"
    im_dir = base / "images"
    js_dir = base / "label_json"
    if not im_dir.exists() or not js_dir.exists():
        return records

    js_files = sorted(js_dir.glob("*.json"))[:limit]
    for jf in tqdm(js_files, desc="RC_10519 Box-Guided"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            img_file = im_dir / (jf.stem + ".jpg")
            if not img_file.exists():
                img_file = im_dir / Path(data.get("imagePath", "")).name
            if not img_file.exists():
                continue

            im = Image.open(img_file).convert("RGB")
            if im.size != (256, 256):
                scale_x = 256.0 / im.width
                scale_y = 256.0 / im.height
                im = im.resize((256, 256), Image.Resampling.LANCZOS)
            else:
                scale_x = 1.0
                scale_y = 1.0

            gray = np.array(im.convert("L"), dtype=np.float32)
            local_mean = uniform_filter(gray, size=17)
            diff = local_mean - gray
            fissure = (diff > 15.0).astype(np.uint8)

            # Build bounding box mask from human annotations
            box_mask = np.zeros((256, 256), dtype=np.uint8)
            for sh in data.get("shapes", []):
                pts = sh.get("points", [])
                if len(pts) >= 2:
                    x1 = int(np.clip(min(pts[0][0], pts[1][0]) * scale_x, 0, 255))
                    x2 = int(np.clip(max(pts[0][0], pts[1][0]) * scale_x, 0, 255))
                    y1 = int(np.clip(min(pts[0][1], pts[1][1]) * scale_y, 0, 255))
                    y2 = int(np.clip(max(pts[0][1], pts[1][1]) * scale_y, 0, 255))
                    box_mask[y1:y2 + 1, x1:x2 + 1] = 1

            masked_crack = fissure * box_mask
            ok, cid, cname, flip_v, meta = classify_structural_mechanism(masked_crack)
            if ok:
                rec = save_sample(
                    im, flip_v,
                    f"rc10519_{jf.stem}",
                    "RC_10519", cid, cname, meta
                )
                records.append(rec)
        except Exception:
            continue
    return records


def process_sdnet_noncracked(limit: int = 3500) -> list[dict]:
    records = []
    p_uncracked = [
        ROOT / "data/raw/SDNET2018/images/D/UD",
        ROOT / "data/raw/SDNET2018/images/P/UP",
        ROOT / "data/raw/SDNET2018/images/W/UW",
    ]
    all_files = []
    for d in p_uncracked:
        if d.exists():
            all_files.extend(sorted(d.glob("*.jpg")))

    step = max(1, len(all_files) // limit)
    selected = all_files[::step][:limit]

    for f in tqdm(selected, desc="SDNET2018 Non-Cracked"):
        try:
            im = Image.open(f).convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.Resampling.LANCZOS)
            rec = save_sample(
                im, False,
                f"sdnet_nc5_{f.name}",
                "SDNET2018", 0, "non_cracked",
                {"angle": 0.0, "length": 0.0, "c_ratio": 0.0, "curvature": 0.0, "taper": 1.0}
            )
            records.append(rec)
        except Exception:
            continue
    return records


def generate_verification_grid(records: list[dict]) -> None:
    """Create a 7-row x 8-column visual contact sheet of real curated samples per mechanism class."""
    VERIFY_GRID.parent.mkdir(parents=True, exist_ok=True)
    cols = 8
    rows = 7
    tile = 128
    pad = 4
    canvas_w = cols * tile + (cols + 1) * pad
    canvas_h = rows * tile + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 28))

    by_class = {cid: [] for cid in range(7)}
    for r in records:
        cid = int(r["class_id"])
        if len(by_class[cid]) < cols:
            by_class[cid].append(ROOT / r["path"])

    for r_idx in range(rows):
        paths = by_class.get(r_idx, [])
        for c_idx, p in enumerate(paths[:cols]):
            if p.exists():
                im = Image.open(p).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                x = pad + c_idx * (tile + pad)
                y = pad + r_idx * (tile + pad)
                canvas.paste(im, (x, y))

    canvas.save(VERIFY_GRID)
    print(f"Saved visual dataset verification grid to: {VERIFY_GRID}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    r_dc = process_deepcrack_all()
    print(f"DeepCrack mechanism samples: {len(r_dc)}")
    all_records.extend(r_dc)

    r_cfd = process_crackforest()
    print(f"CrackForest mechanism samples: {len(r_cfd)}")
    all_records.extend(r_cfd)

    r_rc = process_rc10519_boxed(limit=6000)
    print(f"RC_10519 box-guided mechanism samples: {len(r_rc)}")
    all_records.extend(r_rc)

    r_nc = process_sdnet_noncracked(limit=3500)
    print(f"SDNET2018 non-cracked samples: {len(r_nc)}")
    all_records.extend(r_nc)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "path", "source", "class_id", "class_name",
            "dominant_angle", "max_length", "continuity_ratio",
            "curvature_deg", "taper_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    counts = Counter((int(r["class_id"]), r["class_name"]) for r in all_records)
    print("\n=== Experiment 5 (Level B) Structural Mechanism Dataset Composition ===")
    for (cid, cname), cnt in sorted(counts.items()):
        print(f"  Class {cid} ({cname:20s}): {cnt:5d} native 256x256 images")
    print(f"Total curated dataset: {len(all_records)} images")
    print(f"Saved manifest to: {OUT_CSV}")

    generate_verification_grid(all_records)


if __name__ == "__main__":
    main()
