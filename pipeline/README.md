# Separated Code Pipeline Directory (`pipeline/`)

This directory contains all organized code scripts for **Experiments 1 through 6**, separated from the datasets (`datasets/`) and model weights (`models/weights/`).

---

## 1. One-Command Tester & Image Generator (`pipeline/generate_cracks.py`)

Any user or reviewer can generate images from **any of the 6 experiments** (`--experiment 1..6`), for **any specific crack type** (`--crack-type flexure`, `flexure_shear`, `web_shear`, `torsion`, `corrosion_splitting`, `shrinkage_map`, `non_cracked`, or `all`), and for **any count** (`--count 16`, `200`, `1234`) **without retraining**:

```powershell
cd D:\Downloads\concrete_ddpm

# Generate 1,234 flexural crack images from Experiment 6 (Level C)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type flexure --count 1234 --seed 999

# Generate 200 torsion crack images from Experiment 5 (Level B)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 5 --crack-type torsion --count 200 --seed 2026
```

---

## 2. Subfolder Organization Inside `pipeline/`

- `core/`: Shared diffusion engine (`diffusion.py`), U-Net architectures (`model.py`, `model_attention.py`), dataset loader (`dataset.py`), and analytical structural mechanics solver (`physics_trajectory_solver.py`).
- `exp1_unconditional_64/`: Preprocessing, training, sampling, and evaluation scripts for Experiment 1.
- `exp2_conditional_64/`: Training, conditional sampling, and evaluation scripts for Experiment 2.
- `exp3_conditional_128/`: Dataset preparation, training, and fast DDIM sampling scripts for Experiment 3.
- `exp4_level_a_morphology_256/`: 5-class morphology preparation, `AttentionUNet256` training, DDIM+CFG sampling, and evaluation scripts for Experiment 4.
- `exp5_level_b_mechanisms_256/`: 7-class structural mechanism preparation, class-embedding surgery training, DDIM+CFG sampling, and evaluation scripts for Experiment 5.
- `exp6_level_c_spatial_physics_256/`: Analytical stress-trajectory solver, 5-channel zero-init surgery training, Dual-CFG DDIM sampling, and 1,400-image error analysis scripts for Experiment 6.
