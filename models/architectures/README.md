# Model Architectures (`models/architectures/`)

Contains the PyTorch neural network definitions and Gaussian diffusion process math:
- `model.py`: `SmallUNet` ($64 \times 64$ and $128 \times 128$ U-Net for Experiments 1–3).
- `model_attention.py`: `AttentionUNet256` (`13.82M` parameters, 4 resolution stages `256 -> 128 -> 64 -> 32 -> 16`, SDPA FlashAttention at $32 \times 32$ and $16 \times 16$, Classifier-Free Guidance embedding table `class_embed` with null token).
- `diffusion.py`: `GaussianDiffusion` ($T = 1000$ timesteps, linear $\beta$ schedule).
