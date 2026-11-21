# Separated Datasets & Manifests Directory (`datasets/`)

This directory organizes the training manifests (`train.csv`) and dataset composition documentation for all 6 experiments, separated from the code pipeline (`pipeline/`) and model weights (`models/`).

---

## Master Dataset Comparison Across Experiments 1 to 6

| Subfolder | Experiments Served | Resolution | Total Images | Raw Source Datasets | Classes & Exact Counts |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `exp1_exp2_sdnet_64/` | **Exp 1 & Exp 2** | $64 \times 64$ | **50,482** | `SDNET2018` | `non_cracked`: 42,846<br>`cracked`: 7,636 |
| `exp3_augmented_128/` | **Exp 3** | $128 \times 128$ | **61,001** | `SDNET2018` (50,482)<br>`RC_10519` (10,519) | `non_cracked`: 42,846<br>`cracked`: 18,155 |
| `exp4_level_a_morphology_256/` | **Exp 4 (Level A)** | $256 \times 256$ | **13,139** | `SDNET2018` (11,073)<br>`DeepCrack` (2,066) | `0: non_cracked` (6,000)<br>`1: longitudinal` (978)<br>`2: transverse` (335)<br>`3: diagonal` (686)<br>`4: alligator_map` (5,140) |
| `exp5_exp6_level_bc_mechanisms_256/` | **Exp 5 (Level B) & Exp 6 (Level C)** | $256 \times 256$ | **15,440** | `DeepCrack` (6,344)<br>`RC_10519` (4,832)<br>`SDNET2018` (3,500)<br>`CrackForest` (764) | `0: non_cracked` (3,500)<br>`1: flexural` (944)<br>`2: flexure_shear` (816)<br>`3: web_shear` (2,742)<br>`4: torsion` (1,634)<br>`5: corrosion_splitting` (3,506)<br>`6: shrinkage_map` (2,298) |
