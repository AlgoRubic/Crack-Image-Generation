# Concrete Crack Diffusion Experiments

This directory stores resumable handoff notes and artifacts for each experiment. Raw datasets and shared source code remain in the main project directory.

## Experiments

### Experiment 1: Unconditional DDPM

Path: [unconditional_ddpm/STATUS.md](unconditional_ddpm/STATUS.md)

Purpose: Learn the general distribution of concrete surface images without being told whether an image should be cracked or non-cracked.

Status: Complete baseline.

### Experiment 2: Conditional DDPM

Path: [conditional_ddpm/STATUS.md](conditional_ddpm/STATUS.md)

Purpose: Generate concrete images using an explicit crack-status condition: cracked or non-cracked.

Status: Complete first conditional experiment; crack-specific evaluation still needs improvement.

### Experiment 3: 128x128 Conditional DDPM with Additional Cracked Data

Path: [conditional_ddpm_128_augmented/STATUS.md](conditional_ddpm_128_augmented/STATUS.md)

Purpose: Train the conditional model at 128 x 128 resolution using SDNET2018 plus a genuinely independent cracked-concrete dataset.

Status: Training complete; large-scale evaluation is pending.

Experiment artifacts are stored below the experiment folder in `checkpoints/`, `outputs/`, and `logs/`.

## How to Continue in a New Chat

Start the new chat by saying:

```text
Continue Experiment 1 from D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\STATUS.md
```

or:

```text
Continue Experiment 2 from D:\Downloads\concrete_ddpm\experiments\conditional_ddpm\STATUS.md
```

The new chat should read the selected STATUS.md file first and preserve the existing checkpoints and experiment boundaries. Each experiment folder contains its own artifacts.

## Shared Project Root

```text
D:\Downloads\concrete_ddpm
```

## Shared Dataset

```text
data\raw\SDNET2018\images\
data\processed\
data\splits\
```

## Shared Environment

```text
D:\Downloads\concrete_ddpm\.venv
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
PyTorch: CUDA-enabled
```
