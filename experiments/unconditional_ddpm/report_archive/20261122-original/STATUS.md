# Experiment 1: Unconditional DDPM

## Identity

Experiment name: Unconditional DDPM for Concrete Surface Image Generation

Experiment directory: `D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\`

Project root: `D:\Downloads\concrete_ddpm\`

## Objective

Train a DDPM to generate new 64 x 64 concrete-surface image patches from random noise. The model is not instructed to generate cracked or non-cracked images.

## Dataset

Dataset: SDNET2018

Total indexed images: 56,092

Training images: 50,482

Validation images: 5,610

Surface categories:

```text
bridge_deck
pavement
wall
```

Labels were preserved as metadata but were not supplied to the unconditional model.

Processed data:

```text
data\processed\train\
data\processed\validation\
data\processed\train.csv
data\processed\validation.csv
```

## Model

Architecture: compact U-Net DDPM.

Parameters: 3,258,115.

Image size: 64 x 64 RGB.

Diffusion steps: 1,000.

Loss: mean squared error between actual and predicted Gaussian noise.

Optimizer: AdamW.

Training epochs: 50.

GPU: NVIDIA GeForce RTX 4050 Laptop GPU.

## Checkpoint

Final checkpoint:

```text
checkpoints\ddpm_epoch_050.pt
```

Training log:

```text
outputs\logs\training_loss.csv
```

Final logged loss: `0.01599615436508108`.

## Results

Generated sample grids:

```text
outputs\samples\samples_epoch_050_seed_42.png
outputs\samples\samples_epoch_050_seed_123.png
outputs\samples\samples_epoch_050_seed_999.png
```

Formal evaluation using 1,000 generated and 1,000 real validation images:

```text
FID: 5.604571
KID mean: 3.419779
KID std: 1.028021
Paired LPIPS mean: 0.548679
```

Lightweight diagnostic proxies were also calculated on 2026-09-24 using the
1,000 generated images in `outputs\evaluation\unconditional_epoch_050\` and
the first 1,000 real validation images:

```text
Generated images: 1000
Real comparison images: 1000
Generated mean pixel value: 0.5682
Real mean pixel value: 0.6330
Generated pixel std: 0.1513
Real pixel std: 0.1117
Mean pairwise generated distance: 20.5188
Mean nearest-real pixel distance: 11.1758
```

These are lightweight sanity checks only, not replacements for FID/KID/LPIPS.
The saved diagnostic output is:

```text
outputs\evaluation\unconditional_epoch_050_lightweight_diagnostics.txt
```

Manual audit starter files were created for the first 64 generated evaluation
images:

```text
outputs\audit\unconditional_epoch_050_audit_grid_first_64.png
outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv
```

Detailed full report:

```text
EXPERIMENT_1_FULL_REPORT.md
```

Formula-based PDF report:

```text
reports\experiment_1_unconditional_ddpm_formula_report.pdf
```

## Findings

- The model learned concrete-like color, aggregate texture and surface roughness.
- Crack visibility was inconsistent.
- Because cracked and non-cracked images were mixed without conditioning, the model cannot be instructed to generate a crack.
- The 64 x 64 resolution may remove thin crack details.

## Limitations

- No reliable crack-presence rate was measured for this experiment.
- FID/KID/LPIPS measure general image similarity, not engineering validity or crack correctness.
- The dataset is patch-level and does not contain every structural element or crack cause.
- The 2026-09-24 manual-audit CSV is only a blank annotation template until reviewed by a human.

## Source Files

```text
src/model.py
src/diffusion.py
src/train.py
src/sample.py
src/generate_eval_set.py
src/evaluate_metrics.py
```

## Useful Commands

Generate a sample grid:

```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\Activate.ps1
python .\src\sample.py --checkpoint .\checkpoints\ddpm_epoch_050.pt --count 16 --seed 42
```

## Recommended Continuation

Use this experiment only as the unconditional baseline. The next useful
Experiment 1 step is to manually annotate
`outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv` for visible
crack presence and texture quality, using
`outputs\audit\unconditional_epoch_050_audit_grid_first_64.png` as the review
sheet. For controlled crack generation, continue with Experiment 2 or develop a
mask-conditioned model.
