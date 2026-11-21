# AGENTS.md — `pipeline/exp4_level_a_morphology_256`

- **Scope:** Pipeline scripts for **Experiment 4 (Level A: $256 \times 256$ 5-Class Morphological DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp4_level_a_morphology_256/`
- **Operational Rule:** Loads `models/weights/exp4_level_a_morphology_256/exp4_level_a_morphology_256_epoch050.pt` into `AttentionUNet256(num_classes=5)`.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp4_level_a_morphology_256/`—never re-run training unless explicitly instructed.
