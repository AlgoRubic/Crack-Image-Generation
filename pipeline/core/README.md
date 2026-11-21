# Core Pipeline Modules (`pipeline/core/`)

Shared foundational modules imported across Experiments 1 through 6:

| File | Description |
| :--- | :--- |
| `diffusion.py` | `GaussianDiffusion` ($T = 1000$ linear $\beta_t \in [10^{-4}, 0.02]$ forward noise addition `add_noise` and reverse posterior math). |
| `model.py` | `SmallUNet` architecture used in Experiments 1, 2 ($64 \times 64$) and Experiment 3 ($128 \times 128$). |
| `model_attention.py` | `AttentionUNet256` (`13.82M` parameters) with PyTorch SDPA FlashAttention at $32 \times 32$ and $16 \times 16$ bottlenecks, used in Experiments 4, 5, and 6 ($256 \times 256$). |
| `physics_trajectory_solver.py` | Analytical 2D plane-stress Cauchy + Mohr's Circle + Ornstein-Uhlenbeck aggregate tortuosity + Bernoulli-Euler aperture taper solver for Experiment 6 (Level C). |
| `dataset.py` | Baseline CSV image dataset loader for Experiments 1–3. |
| `evaluate_exp1_2_3_errors.py` | Unified error and confusion-matrix evaluator for Experiments 1, 2, and 3. |
