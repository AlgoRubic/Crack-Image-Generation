# Experiment 1: Baseline Unconditional DDPM ($64 \times 64$) — Tester & Replication Guide

## 1. Overview of Experiment 1
- **Objective:** Train an unconditional Denoising Diffusion Probabilistic Model (`SmallUNet`, $T = 1000$ linear noise schedule) to synthesize realistic $64 \times 64$ concrete surface images from random Gaussian noise.
- **Dataset Used:** `SDNET2018` (`50,482` processed $64 \times 64$ concrete images in `data/processed/train.csv`: `42,846` non-cracked and `7,636` cracked across bridge decks, pavements, and walls).
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp1_unconditional_64/exp1_unconditional_64_epoch050.pt` (Epoch 50, 37.4 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 1 Sampler**:

### Option A: Using the Unified Generator (Recommended — Saves Individual PNGs + Preview Grid)
```powershell
cd D:\Downloads\concrete_ddpm

# Generate 16 unconditional 64x64 concrete images (replicates report seed 2026)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 1 --count 16 --seed 2026

# Generate any custom number of new images (e.g., 100 or 1234 images) with any seed
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 1 --count 100 --seed 999
```
- **Where Your Generated Images Are Saved:**
  `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp1\unconditional\`

### Option B: Using the Standalone Experiment 1 Script (`sample.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp1_unconditional_64/sample.py `
  --checkpoint models/weights/exp1_unconditional_64/exp1_unconditional_64_epoch050.pt `
  --count 16 `
  --seed 42
```

---

## 3. Files Included in This Folder (`exp1_unconditional_64/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `sample.py` | **Inference / Testing Script:** Loads the pre-trained checkpoint and generates $64 \times 64$ concrete image grids using 1,000-step reverse DDPM sampling. |
| `evaluate_baseline.py` | **Evaluation Script:** Computes baseline pixel statistics, crack presence rate, and distribution metrics on generated Experiment 1 images. |
| `preprocess_images.py` | **Data Preparation Reference:** Shows how the raw `SDNET2018` dataset was resized to $64 \times 64$ via Lanczos resampling (`data/processed/train.csv`). |
| `train.py` | **Training Reference (Not needed for testing):** The script used to train `SmallUNet` for 50 epochs (`ddpm_epoch_050.pt`). |
