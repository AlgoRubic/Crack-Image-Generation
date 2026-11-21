# AGENTS.md — `pipeline/exp6_level_c_spatial_physics_256`

- **Scope:** Pipeline scripts for **Experiment 6 (Level C: $256 \times 256$ Spatial-Physics Trajectory & Aperture DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp6_level_c_spatial_physics_256/`
- **Operational Rule:** Loads `models/weights/exp6_level_c_spatial_physics_256/exp6_level_c_spatial_physics_256_epoch015.pt` into 5-channel `AttentionUNet256(in_channels=5, out_channels=3, num_classes=7)`.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp6_level_c_spatial_physics_256/`—never re-run training unless explicitly instructed.
