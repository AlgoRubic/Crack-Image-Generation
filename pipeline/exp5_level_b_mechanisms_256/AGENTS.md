# AGENTS.md — `pipeline/exp5_level_b_mechanisms_256`

- **Scope:** Pipeline scripts for **Experiment 5 (Level B: $256 \times 256$ 7-Class Structural Failure Mechanism DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp5_level_b_mechanisms_256/`
- **Operational Rule:** Loads `models/weights/exp5_level_b_mechanisms_256/exp5_level_b_mechanisms_256_epoch025.pt` into `AttentionUNet256(num_classes=7)`.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp5_level_b_mechanisms_256/`—never re-run training unless explicitly instructed.
