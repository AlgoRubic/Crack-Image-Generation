# Experiment 5 (Level B): 7-Class Structural Failure Mechanism DDPM ($256 \times 256$) — Tester & Replication Guide

## 1. Overview of Experiment 5 (Level B)
- **Objective:** Advance from 2D geometric crack angles (Level A) to **7 Reinforced-Concrete Structural Failure Mechanisms (Level B)** at native **$256 \times 256$ resolution**, using Warm-Start Class-Embedding Surgery (`6 -> 8 tokens`, transferring 100% of Experiment 4's U-Net backbone for 25 additional epochs = **75 cumulative epochs**):
  - `0: non_cracked` — Intact uncracked concrete surface
  - `1: flexural` (or `flexure`) — Bottom-initiated vertical bending crack tapering upward
  - `2: flexure_shear` — Vertical flexural crack at bottom curving diagonally ($\sim 45^\circ / 135^\circ$) into the web
  - `3: web_shear` — Single dominant inclined diagonal tension crack in the beam web
  - `4: torsion` — Parallel echelon multi-crack diagonal spiral fractures across the member face
  - `5: corrosion_splitting` — Continuous horizontal splitting crack along longitudinal rebar
  - `6: shrinkage_map` — Isotropic polygonal drying/plastic shrinkage crazing network
- **Dataset Used:** **`15,440` native $256 \times 256$ images** in `data/processed_exp5_mechanisms/train.csv` compiled from **4 raw sources**:
  - `DeepCrack` (`6,344` pixel-mask verified crops)
  - `RC_10519` (`4,832` human JSON bounding-box guided images)
  - `SDNET2018` (`3,500` non-cracked concrete tiles)
  - `CrackForest (CFD)` (`764` ground-truth `.mat` mask crops)
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp5_level_b_mechanisms_256/exp5_level_b_mechanisms_256_epoch025.pt` (Epoch 25 / Cumulative Epoch 75, 158.4 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 5 Sampler**:

### Option A: Using the Unified Generator (Recommended — Generate Any Specific Failure Mechanism or Any Count Like `1234`)
```powershell
cd D:\Downloads\concrete_ddpm

# 1. Generate 1,234 NEW FLEXURAL crack images (accepts "flexure" or "flexural")
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 5 --crack-type flexure --count 1234 --seed 999

# 2. Generate any other specific structural mechanism (e.g. torsion, flexure_shear, web_shear, corrosion_splitting, shrinkage_map, non_cracked)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 5 --crack-type torsion --count 25 --seed 2026

# 3. Replicate all 7 structural failure mechanisms (16 images per class = 112 images + 4x4 grids)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 5 --crack-type all --count 16 --seed 2026
```
- **Where Your Generated Images Are Saved:**
  `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp5\<crack_type>\`

### Option B: Using the Standalone Experiment 5 Script (`sample_exp5_mechanisms_256.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp5_level_b_mechanisms_256/sample_exp5_mechanisms_256.py `
  --checkpoint models/weights/exp5_level_b_mechanisms_256/exp5_level_b_mechanisms_256_epoch025.pt `
  --count 16 `
  --steps 50 `
  --cfg-scale 2.5 `
  --seed 2026 `
  --output-dir replicate_and_generate_suite/generated_outputs/exp5_standalone
```

---

## 3. Files Included in This Folder (`exp5_level_b_mechanisms_256/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `sample_exp5_mechanisms_256.py` | **Inference / Testing Script:** Fast 50-step DDIM + CFG ($w=2.5$) sampler for all 7 structural failure mechanism classes at native $256 \times 256$. |
| `evaluate_exp5_mechanisms_256.py` | **Evaluation Script:** Computes per-class crack presence, structural mechanism adherence, piecewise curvature ($\Delta\theta$), EDT aperture width taper ($r_{\text{taper}}$), and class-matched FID/KID (`KID = 0.03931`). |
| `prepare_experiment5_mechanisms.py` | **Data Preparation Reference:** Classifies `DeepCrack`, `CrackForest`, `RC_10519`, and `SDNET2018` into the 7 structural failure mechanisms using 5 mechanics descriptors and canonical bottom-tension alignment. |
| `train_exp5_mechanisms_256.py` | **Training Reference (Not needed for testing):** Performs analytical class-embedding surgery (`6 -> 8 tokens`) from Experiment 4 and fine-tunes for 25 epochs. |
