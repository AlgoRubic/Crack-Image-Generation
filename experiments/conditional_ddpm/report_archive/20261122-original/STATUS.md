# Experiment 2: Conditional DDPM

## Identity

Experiment name: Crack-Status Conditional DDPM for Concrete Images

Experiment directory: `D:\Downloads\concrete_ddpm\experiments\conditional_ddpm\`

Project root: `D:\Downloads\concrete_ddpm\`

## Objective

Train a DDPM that receives a binary crack-status condition and can generate separate cracked and non-cracked concrete image sets.

Conditions:

```text
class 0 = non_cracked
class 1 = cracked
```

## Dataset

Dataset: same SDNET2018 processed dataset as Experiment 1.

Training images: 50,482.

Validation images: 5,610.

The conditional training script uses a balanced sampler so the cracked class is not overwhelmed by the larger non-cracked class.

## Model

Architecture: same compact U-Net as Experiment 1 plus a learned two-class embedding.

Parameters: 3,258,627.

Additional parameters compared with Experiment 1: 512.

Image size: 64 x 64 RGB.

Diffusion steps: 1,000.

Loss: mean squared error between actual and predicted Gaussian noise.

Optimizer: AdamW.

Training epochs: 50.

GPU: NVIDIA GeForce RTX 4050 Laptop GPU.

## Checkpoint

Final checkpoint:

```text
checkpoints\conditional\conditional_epoch_050.pt
```

Training log:

```text
outputs\logs\conditional_training_loss.csv
```

Final loss from the plotted comparison: approximately `0.015247`.

## Generated Evaluation Sets

```text
outputs\evaluation\conditional_cracked\
outputs\evaluation\conditional_non_cracked\
```

Each contains 1,000 individual images.

Sample grids:

```text
outputs\conditional_samples\cracked_epoch_050_seed_42.png
outputs\conditional_samples\non_cracked_epoch_050_seed_42.png
outputs\conditional_samples\cracked_epoch_050_seed_123.png
outputs\conditional_samples\non_cracked_epoch_050_seed_123.png
```

## Results

Metrics were calculated against the same 1,000-image real validation comparison set.

### Cracked condition

```text
FID: 0.701002
KID mean: 0.556873
KID std: 0.202802
Paired LPIPS mean: 0.425783
```

### Non-cracked condition

```text
FID: 0.724640
KID mean: 0.756606
KID std: 0.389974
Paired LPIPS mean: 0.416435
```

Detailed formula-based PDF report:

```text
reports\experiment_2_conditional_ddpm_formula_report.pdf
```

## Findings

- The conditional model has substantially lower FID and KID than the unconditional baseline under the same evaluation procedure.
- A preliminary manual inspection found approximately 5 visibly cracked images out of 16 in one cracked-condition grid, approximately 31.25%.
- This 16-image observation is not a final crack-presence metric.
- Crack-specific automatic evaluation is currently blocked because the first auxiliary classifier reached only 65.79% validation accuracy.

## Important Evaluation Status

Do not claim that the conditional model reliably generates cracks based only on FID/KID/LPIPS. These metrics measure general image-distribution similarity, not crack correctness.

The crack-presence rate remains pending until one of these is completed:

1. Manual annotation of a representative generated sample.
2. A stronger validated crack classifier with precision, recall and F1.

## Source Files

```text
src/model.py
src/diffusion.py
src/train_conditional.py
src/sample_conditional.py
src/generate_conditional_eval_set.py
src/evaluate_metrics.py
src/train_crack_classifier.py
src/measure_crack_presence.py
```

## Useful Commands

Generate a cracked grid:

```powershell
cd D:\Downloads\concrete_ddpm
.\.venv\Scripts\Activate.ps1
python .\src\sample_conditional.py --checkpoint .\checkpoints\conditional\conditional_epoch_050.pt --class-name cracked --count 16 --seed 42
```

Generate a non-cracked grid:

```powershell
python .\src\sample_conditional.py --checkpoint .\checkpoints\conditional\conditional_epoch_050.pt --class-name non_cracked --count 16 --seed 42
```

Recalculate FID/KID/LPIPS for the cracked condition:

```powershell
python .\src\evaluate_metrics.py --real .\data\processed\validation --fake .\outputs\evaluation\conditional_cracked --batch-size 32 --max-images 1000
```

## Recommended Continuation

First complete crack-presence evaluation. If the rate remains weak, test a stronger evaluator and then compare a 128 x 128 conditional model or a mask-conditioned diffusion model.
