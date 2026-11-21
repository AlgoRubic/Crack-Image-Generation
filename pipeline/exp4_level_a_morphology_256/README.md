# Experiment 4 (Level A): 5-Class Morphological Crack-Type Conditioned DDPM ($256 \times 256$) — Tester & Replication Guide

## 1. Overview of Experiment 4 (Level A)
- **Objective:** Upgrade to **native $256 \times 256$ resolution** using a **13.82M-parameter `AttentionUNet256`** with PyTorch SDPA FlashAttention ($32 \times 32$ and $16 \times 16$ bottlenecks), Classifier-Free Guidance (CFG), and fast **50-step DDIM sampling**, conditioned on **5 geometric crack morphology classes**:
  - `0: non_cracked` — Clean intact concrete surface
  - `1: longitudinal` — Horizontal crack ($\theta \approx 0^\circ \pm 22.5^\circ$)
  - `2: transverse` — Vertical crack ($\theta \approx 90^\circ \pm 22.5^\circ$)
  - `3: diagonal` — Inclined shear-angle crack ($\theta \approx 45^\circ$ or $135^\circ \pm 22.5^\circ$)
  - `4: alligator_map` — Multi-branch interconnected map cracking ($\ge 4$ branches)
- **Dataset Used:** **`13,139` native $256 \times 256$ images** in `data/processed_256/train.csv` (`SDNET2018`: `11,073` images + `DeepCrack`: `2,066` high-resolution sliding-window crops), filtered for verified crack continuity ($L_{\max} \ge 50\text{ px}$).
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp4_level_a_morphology_256/exp4_level_a_morphology_256_epoch050.pt` (Epoch 50, 158.4 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 4 Sampler**:

### Option A: Using the Unified Generator (Recommended — Generate Any Single Crack Type or All 5 Classes)
```powershell
cd D:\Downloads\concrete_ddpm

# 1. Replicate all 5 classes (16 images per class = 80 images + 4x4 grids) using report seed 2026
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 4 --crack-type all --count 16 --seed 2026

# 2. Generate ONLY ONE specific geometric crack type (e.g. diagonal, longitudinal, transverse, alligator_map, non_cracked)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 4 --crack-type diagonal --count 50 --seed 2026

# 3. Generate 1,234 images of alligator_map or transverse cracks with a new seed
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 4 --crack-type alligator_map --count 1234 --seed 999
```
- **Where Your Generated Images Are Saved:**
  `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp4\<crack_type>\`

### Option B: Using the Standalone Experiment 4 Script (`sample_crack_type_256.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp4_level_a_morphology_256/sample_crack_type_256.py `
  --checkpoint models/weights/exp4_level_a_morphology_256/exp4_level_a_morphology_256_epoch050.pt `
  --count 16 `
  --steps 50 `
  --cfg-scale 2.5 `
  --seed 2026 `
  --output-dir replicate_and_generate_suite/generated_outputs/exp4_standalone
```

---

## 3. Files Included in This Folder (`exp4_level_a_morphology_256/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `sample_crack_type_256.py` | **Inference / Testing Script:** Fast 50-step DDIM + CFG ($w=2.5$) sampler for all 5 morphological crack classes at native $256 \times 256$. |
| `evaluate_crack_type_256.py` | **Evaluation Script:** Evaluates per-class crack presence rate, 2nd-order central-moment orientation adherence, continuity ratio, and FID/KID (`KID = 0.01105`). |
| `prepare_experiment4_256.py` | **Data Preparation Reference:** Automated central-moment inertia tensor and multi-tier continuity classifier that built the 5-class dataset (`data/processed_256/train.csv`). |
| `train_crack_type_256.py` | **Training Reference (Not needed for testing):** The 50-epoch FP16 AMP training script for `AttentionUNet256`. |
