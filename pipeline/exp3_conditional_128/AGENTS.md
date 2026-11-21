# AGENTS.md — `pipeline/exp3_conditional_128`

- **Scope:** Pipeline scripts for **Experiment 3 ($128 \times 128$ Augmented Binary Conditional DDPM)**.
- **Pre-Trained Weight Location:** `models/weights/exp3_conditional_128/`
- **Operational Rule:** Loads `models/weights/exp3_conditional_128/exp3_conditional_128_epoch050.pt`. Use `sample_exp3_128_fast.py` for fast 50-step DDIM sampling.
- **Testing vs. Training:** When asked to test, sample, or replicate results, always run the sampling/generation script or `pipeline/generate_cracks.py` using the pre-trained `.pt` checkpoint in `models/weights/exp3_conditional_128/`—never re-run training unless explicitly instructed.
