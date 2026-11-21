# Separated Models & Pre-Trained Weights Directory (`models/`)

This directory isolates all **neural network model architectures** (`models/architectures/`) and **final pre-trained model weights (`.pt`)** (`models/weights/`) from the datasets (`datasets/`) and execution scripts (`pipeline/`).

---

## Inventory of Pre-Trained Model Weights (`models/weights/`)

| Experiment | Weight Subfolder | Checkpoint Filename | Architecture | Resolution | Size |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Exp 1** | `weights/exp1_unconditional_64/` | `exp1_unconditional_64_epoch050.pt` | `SmallUNet(in=3, num_classes=None)` | $64 \times 64$ | 37.4 MB |
| **Exp 2** | `weights/exp2_conditional_64/` | `exp2_conditional_64_epoch050.pt` | `SmallUNet(in=3, num_classes=2)` | $64 \times 64$ | 12.5 MB |
| **Exp 3** | `weights/exp3_conditional_128/` | `exp3_conditional_128_epoch050.pt` | `SmallUNet(in=3, num_classes=2)` | $128 \times 128$ | 12.5 MB |
| **Exp 4 (Level A)** | `weights/exp4_level_a_morphology_256/` | `exp4_level_a_morphology_256_epoch050.pt` | `AttentionUNet256(in=3, num_classes=5)` | $256 \times 256$ | 158.4 MB |
| **Exp 5 (Level B)** | `weights/exp5_level_b_mechanisms_256/` | `exp5_level_b_mechanisms_256_epoch025.pt` | `AttentionUNet256(in=3, num_classes=7)` | $256 \times 256$ | 158.4 MB |
| **Exp 6 (Level C)** | `weights/exp6_level_c_spatial_physics_256/` | `exp6_level_c_spatial_physics_256_epoch015.pt` | `AttentionUNet256(in=5, out=3, num_classes=7)` | $256 \times 256$ | 158.4 MB |
