# AGENTS.md — `pipeline/exp2_conditional_64`

- **Scope:** Pipeline scripts for **Experiment 2 ($64 \times 64$ Binary Conditional DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp2_conditional_64/`
- **Operational Rule:** Loads `models/weights/exp2_conditional_64/exp2_conditional_64_epoch050.pt`. Supports `0: non_cracked` and `1: cracked`.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp2_conditional_64/`—never re-run training unless explicitly instructed.
