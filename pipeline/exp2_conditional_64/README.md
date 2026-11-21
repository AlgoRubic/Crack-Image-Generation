# Experiment 2: Binary Class-Conditioned DDPM ($64 \times 64$) — Tester & Replication Guide

## 1. Overview of Experiment 2
- **Objective:** Introduce binary class conditioning (`0: non_cracked`, `1: cracked`) to `SmallUNet` at $64 \times 64$ resolution so the user can explicitly choose whether to generate a **clean intact concrete surface** or a **cracked concrete surface**.
- **Dataset Used:** `SDNET2018` (`50,482` images in `data/processed/train.csv`: `42,846` `non_cracked` and `7,636` `cracked`, balanced during training via `WeightedRandomSampler`).
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp2_conditional_64/exp2_conditional_64_epoch050.pt` (Epoch 50, 12.5 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 2 Sampler**:

### Option A: Using the Unified Generator (Recommended — Generate Any Count for `cracked`, `non_cracked`, or `all`)
```powershell
cd D:\Downloads\concrete_ddpm

# 1. Generate 16 CRACKED concrete images
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 2 --crack-type cracked --count 16 --seed 2026

# 2. Generate 16 NON-CRACKED (clean) concrete images
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 2 --crack-type non_cracked --count 16 --seed 2026

# 3. Generate both classes at once, or generate 500+ images of cracked concrete
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 2 --crack-type all --count 16 --seed 2026
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 2 --crack-type cracked --count 500 --seed 123
```
- **Where Your Generated Images Are Saved:**
  - `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp2\cracked\`
  - `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp2\non_cracked\`

### Option B: Using the Standalone Experiment 2 Script (`sample_conditional.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp2_conditional_64/sample_conditional.py `
  --checkpoint models/weights/exp2_conditional_64/exp2_conditional_64_epoch050.pt `
  --class-name cracked `
  --count 16 `
  --seed 42
```

---

## 3. Files Included in This Folder (`exp2_conditional_64/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `sample_conditional.py` | **Inference / Testing Script:** Generates either `cracked` (`1`) or `non_cracked` (`0`) $64 \times 64$ concrete images from the pre-trained checkpoint. |
| `evaluate_metrics.py` | **Evaluation Script:** Evaluates conditional class fidelity, crack presence rates, and FID/KID metrics on Experiment 2 outputs. |
| `train_conditional.py` | **Training Reference (Not needed for testing):** The class-balanced conditional training script used to produce `conditional_epoch_050.pt`. |
