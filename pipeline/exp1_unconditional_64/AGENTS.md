# AGENTS.md — `pipeline/exp1_unconditional_64`

- **Scope:** Pipeline scripts for **Experiment 1 ($64 \times 64$ Unconditional DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp1_unconditional_64/`
- **Operational Rule:** Loads `models/weights/exp1_unconditional_64/exp1_unconditional_64_epoch050.pt`. Use `sample.py` or `pipeline/generate_cracks.py --experiment 1`.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp1_unconditional_64/`—never re-run training unless explicitly instructed.
