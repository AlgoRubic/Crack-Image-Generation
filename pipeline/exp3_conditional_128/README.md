# Experiment 3: Augmented High-Count Binary Conditional DDPM ($128 \times 128$) — Tester & Replication Guide

## 1. Overview of Experiment 3
- **Objective:** Double the spatial resolution from $64 \times 64$ to **$128 \times 128$** ($4\times$ more pixels) and augment the training set with `10,519` additional reinforced-concrete crack photographs (`RC_10519`) to increase crack diversity.
- **Dataset Used:** Combined `SDNET2018` (`50,482`) + `RC_10519` (`10,519`) = **`61,001` total $128 \times 128$ images** in `data/processed_128/train.csv` (`42,846` `non_cracked` + `18,155` `cracked`).
- **Pre-Trained Checkpoint Included (No Retraining Required):**
  `models/weights/exp3_conditional_128/exp3_conditional_128_epoch050.pt` (Epoch 50, 12.5 MB)

---

## 2. How to Generate Images Immediately (Testing Only — No Retraining Needed)

Open PowerShell inside `D:\Downloads\concrete_ddpm` and run either the **Unified Generator** or the **Standalone Experiment 3 Generator**:

### Option A: Using the Unified Generator (Recommended — Generate Any Count for `cracked`, `non_cracked`, or `all`)
```powershell
cd D:\Downloads\concrete_ddpm

# 1. Generate 16 CRACKED 128x128 concrete images
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 3 --crack-type cracked --count 16 --seed 2026

# 2. Generate 16 NON-CRACKED 128x128 concrete images
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 3 --crack-type non_cracked --count 16 --seed 2026

# 3. Generate any custom number of images (e.g. 200 cracked images)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 3 --crack-type cracked --count 200 --seed 999
```
- **Where Your Generated Images Are Saved:**
  - `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp3\cracked\`
  - `D:\Downloads\concrete_ddpm\replicate_and_generate_suite\generated_outputs\exp3\non_cracked\`

### Option B: Using the Standalone Experiment 3 Script (`generate_eval_128.py`)
```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\python.exe replicate_and_generate_suite/exp3_conditional_128/generate_eval_128.py `
  --checkpoint models/weights/exp3_conditional_128/exp3_conditional_128_epoch050.pt `
  --class-name cracked `
  --count 16 `
  --batch-size 16 `
  --seed 2026 `
  --output-dir replicate_and_generate_suite/generated_outputs/exp3_standalone
```

---

## 3. Files Included in This Folder (`exp3_conditional_128/`)

| File Name | Role & What the Tester Can Do With It |
| :--- | :--- |
| `generate_eval_128.py` | **Inference / Testing Script:** Generates individual $128 \times 128$ `cracked` or `non_cracked` PNG images with deterministic resume support. |
| `prepare_experiment3_128.py` | **Data Preparation Reference:** Combines `SDNET2018` (`50,482`) and `RC_10519` (`10,519`) into the `61,001`-image $128 \times 128$ dataset. |
| `train_conditional_128_augmented.py` | **Training Reference (Not needed for testing):** The script used to train the 50-epoch Experiment 3 model. |
