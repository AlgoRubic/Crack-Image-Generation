# Experiment 6 (Level C): Physics-Guided Spatial Trajectory & Aperture-Field Conditioned DDPM ($256 \times 256$) — Tester & Replication Guide

## 1. Overview of Experiment 6 (Level C)
- **Objective:** Achieve **100% physical trajectory and strain-compatible aperture width taper compliance** across all **7 Reinforced-Concrete Structural Failure Mechanisms** at native **$256 \times 256$ resolution**, while eliminating unnatural camera white-balance color shifts (neutral Portland cement grey concrete coloration).
- **How Level C Works:**
  1. **Analytical Structural Mechanics Solver (`physics_trajectory_solver.py`):** Solves 2D plane-stress Cauchy equilibrium, Mohr's Circle principal tensile stress trajectories ($\theta_{\text{crack}}(x,y) = \alpha_1(x,y) + \pi/2$), Ornstein-Uhlenbeck coarse-aggregate tortuosity SDE, and Bernoulli-Euler / Bredt-Batho strain-compatible aperture width profiles to generate a 2-channel spatial physics conditioning field $\mathbf{S}(x, y) = [S_{\text{skeleton}}, S_{\text{aperture}}] \in [-1, 1]^{2 \times 256 \times 256}$.
  2. **5-Channel `PhysicsControlUNet256` (`13,821,043` parameters):** Created via Zero-Initialized Input-Channel Surgery (`3 -> 5 channels`) from Experiment 5 and fine-tuned for **15 epochs** (**90 cumulative epochs** across Experiments 4–6), achieving an all-time best MSE loss of **`0.011051`**.
- **7 Structural Failure Mechanism Classes Supported:**
  - `0: non_cracked` — Clean uncracked concrete surface (`100.0%` clean rate)
  - `1: flexural` (or `flexure`) — Bottom-initiated vertical bending crack with Bernoulli-Euler upward aperture width taper ($r_{\text{taper}} = 1.50$)
  - `2: flexure_shear` — Bottom-initiated vertical crack rotating smoothly into a $45^\circ / 135^\circ$ diagonal web-shear J-curve ($\Delta\theta = 37.1^\circ, r_{\text{taper}} = 1.32$)
  - `3: web_shear` — Diagonal tension fracture centered in the web with parabolic mid-depth aperture profile
  - `4: torsion` — 2 to 3 parallel echelon diagonal fractures from circulatory Bredt-Batho shear flow
  - `5: corrosion_splitting` — Continuous straight horizontal splitting fracture along the rebar axis ($\Delta\theta = 0.3^\circ$)
  - `6: shrinkage_map` — Interconnected Voronoi polygonal drying-shrinkage crazing network with $120^\circ$ triple junctions
- **Dataset Used:** **`15,440` native $256 \times 256$ images** in `data/processed_exp5_mechanisms/train.csv` (`DeepCrack`: `6,344`, `RC_10519`: `4,832`, `SDNET2018`: `3,500`, `CrackForest`: `764`), paired with on-the-fly `[S_skeleton, S_aperture]` extraction and chromatic white-balance normalization.
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp6_level_c_spatial_physics_256/exp6_level_c_spatial_physics_256_epoch015.pt` (Epoch 15 / Cumulative Epoch 90, 158.4 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 6 Sampler**:

### Option A: Using the Unified Generator (Recommended — Generate Any Specific Crack Type or Any Count Like `200` or `1234`)
```powershell
cd D:\Downloads\concrete_ddpm

# 1. Generate 1,234 NEW physics-controlled FLEXURAL crack images (accepts "flexure" or "flexural")
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type flexure --count 1234 --seed 999

# 2. Generate 200 physics-controlled FLEXURE-SHEAR (J-curve) or TORSION crack images
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type flexure_shear --count 200 --seed 2026

# 3. Replicate all 7 classes (16 images per class = 112 images + paired .npy physics masks + 4x4 grids)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type all --count 16 --seed 2026
```
- **Where Your Generated Images & Paired Physics Masks Are Saved:**
  - Images: `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp6\<crack_type>\`
  - Paired Physics Masks: `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp6\<crack_type>_physics_masks\`

### Option B: Using the Standalone Experiment 6 Script (`sample_exp6_spatial_physics_256.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp6_level_c_spatial_physics_256/sample_exp6_spatial_physics_256.py `
  --checkpoint models/weights/exp6_level_c_spatial_physics_256/exp6_level_c_spatial_physics_256_epoch015.pt `
  --count 16 `
  --steps 50 `
  --cfg-scale 2.5
```

---

## 3. Files Included in This Folder (`exp6_level_c_spatial_physics_256/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `physics_trajectory_solver.py` | **Analytical Structural Mechanics Solver:** Generates physically valid 2-channel `[S_skeleton, S_aperture]` fields for any of the 7 failure mechanisms (`0..6`). Run it directly (`python physics_trajectory_solver.py`) to generate a $7 \times 8$ visual verification grid of analytical trajectories! |
| `sample_exp6_spatial_physics_256.py` | **Inference / Testing Script:** Runs 50-step DDIM + Dual-CFG ($w=2.5$) guided by `physics_trajectory_solver.py` to synthesize native $256 \times 256$ images and paired `.npy` physics masks. |
| `evaluate_exp6_spatial_physics_256.py` | **Evaluation & Error Analysis Script:** Evaluates generated images against their paired `.npy` analytical physics masks (`100.0%` mechanism adherence, `98.9%` spatial trajectory recall, curvature $\Delta\theta$, width taper $r_{\text{taper}}$, and chromatic deviation). |
| `train_exp6_spatial_physics_256.py` | **Training Reference (Not needed for testing):** Performs zero-initialized input-channel surgery (`3 -> 5 channels`) from Experiment 5 and trains `PhysicsControlUNet256` for 15 epochs. |
