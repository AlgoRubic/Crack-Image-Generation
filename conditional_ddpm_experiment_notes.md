# Conditional DDPM Experiment Notes

This file belongs only to the next experiment. The unconditional baseline is documented in `experiment_notes.md`.

## Objective

Generate concrete images with explicit control over crack status:

```text
class 0 = non-cracked
class 1 = cracked
```

The main question is whether adding a crack-status condition increases the proportion and clarity of visible cracks in generated images.

## Why This Experiment Is Needed

The unconditional DDPM generated realistic concrete texture but did not reliably generate visible cracks. The training data contains both labels, while the model receives no label during training or sampling.

## Planned Model Change

Add a learned class embedding to the timestep embedding:

```text
time embedding + class embedding -> residual U-Net blocks
```

Training input:

```text
noisy image + timestep + crack-status label
```

Sampling input:

```text
random noise + timestep + requested crack-status label
```

## Planned Experiments

### C1: Conditional 64 x 64

- Same dataset split as the baseline.
- Same 1,000-step diffusion schedule.
- Same U-Net capacity initially.
- Add the crack-status label.
- Generate equal numbers of cracked and non-cracked samples.

### C2: Conditional 128 x 128

Run only if C1 still produces unclear crack structures. This tests whether resolution is the limiting factor after conditioning is added.

## Evaluation Plan

Compare the unconditional baseline and conditional model using visual grids, crack-presence rate, FID/KID when a suitable feature implementation is available, sample diversity, nearest-real-image similarity, downstream utility and memorization checks.

## Success Criteria

The conditional model should generate visibly different cracked and non-cracked groups, increase crack-presence rate for the cracked condition, preserve realistic concrete texture, avoid identical samples and remain reproducible across random seeds.

## Current Status

```text
Baseline evaluation proxies: pending
Conditional U-Net implementation: pending
Conditional training: pending
Conditional sampling: pending
```

## Implementation Started

The conditional U-Net now receives a learned binary class embedding. Training uses balanced sampling so the cracked class is not overwhelmed by the larger non-cracked class.

Files:

```text
src/train_conditional.py
src/sample_conditional.py
```

The original unconditional model and its checkpoint remain unchanged.

## Training Run Status

The 50-epoch conditional DDPM training run has been started with the balanced class sampler, CUDA mixed precision and the same 64 x 64 image resolution as the baseline.

Expected outputs:

```text
checkpoints/conditional/conditional_epoch_001.pt ... conditional_epoch_050.pt
outputs/logs/conditional_training_loss.csv
```

After training, generate separate `cracked` and `non_cracked` sample grids using the epoch-50 checkpoint and compare them with the unconditional baseline.

## Initial Conditional Sampling Observation

The generated `cracked` grid contained approximately 5 visibly cracked images out of 16 samples, or about 31.25% by manual inspection.

### Interpretation

This is an improvement in direction over the unconditional baseline if the corresponding non-cracked grid contains fewer visible cracks, but it is not reliable crack control. A successful conditional model should produce a substantially higher crack-presence rate for the `cracked` condition and a substantially lower rate for the `non_cracked` condition.

### Caution

Sixteen samples are too few for a stable percentage. The 31.25% value is a preliminary manual observation, not a final metric.

### Next diagnostic

Generate at least 100 cracked and 100 non-cracked samples with fixed seeds and manually record:

```text
cracked condition: visible cracked / 100
non-cracked condition: visible cracked / 100
unclear samples: count
```

If the separation remains weak, test crack-focused training data, 128 x 128 resolution or stronger conditioning before claiming success.

## Crack-Presence Evaluation

A separate binary classifier is used to estimate whether generated images visibly contain cracks. It is trained only on real SDNET2018 labels and evaluated on the held-out real validation set before being applied to generated images.

Files:

```text
src/train_crack_classifier.py
src/measure_crack_presence.py
```

This classifier-based rate is an estimate, not a replacement for expert inspection. Its validation accuracy and confusion matrix must be recorded before trusting the generated crack-presence rates.

### Classifier validation result

The initial classifier run was stopped because its validation accuracy was too low for reliable generated-image evaluation:

```text
Epoch 1 validation accuracy: 0.5645
Epoch 2 validation accuracy: 0.6579
Epoch 1 confusion matrix:
[[2507, 2255],
 [ 188,  660]]
Epoch 2 confusion matrix:
[[3086, 1676],
 [ 243,  605]]
```

The classifier is not yet trusted for crack-presence measurement. The generated-image crack-presence rate must remain reported as pending until a stronger classifier reaches an acceptable held-out validation performance, preferably with per-class precision, recall and F1 also reported.

## Detailed Next-Phase Plan

### Phase N1: Manual crack-presence audit

Manually label a stratified sample of generated images before training another generator:

```text
200 cracked-condition images
200 non-cracked-condition images
```

Use three labels:

```text
1 = clearly visible crack
0 = clearly no crack
-1 = unclear or ambiguous
```

Report the crack-presence rate after excluding unclear images and report the unclear count separately.

### Phase N2: Improve the automatic evaluator

Train a stronger transfer-learning classifier, such as ResNet18 or EfficientNet, using the real-only train/validation split, class-balanced sampling and augmentation. Report accuracy, cracked-class precision, cracked-class recall, cracked-class F1, balanced accuracy and a confusion matrix.

Do not apply the classifier to generated images until its held-out real validation performance is strong and its errors have been visually inspected.

### Phase N3: Recalculate crack-presence rate

Apply the validated classifier to 1,000 unconditional images, 1,000 conditional cracked images and 1,000 conditional non-cracked images. Compare automatic rates with the manual audit. If they disagree substantially, use manual labels in the final report.

### Phase N4: Decide whether to change the generator

```text
If conditional cracked rate is clearly higher than non-cracked rate:
    keep the conditional DDPM and evaluate downstream utility.

If both rates are similar and low:
    improve conditioning and use crack-focused training data.

If cracks remain faint:
    repeat the conditional experiment at 128 x 128.
```

### Phase N5: 128 x 128 conditional experiment

Begin this only after the 64 x 64 evaluation is complete. Repeat the same training protocol at 128 x 128, reduce batch size if necessary, and use the same evaluation procedure.

### Phase N6: Downstream usefulness test

Train a crack classifier or segmentation model using:

```text
real training images only
real images plus selected synthetic images
```

Evaluate both on a fixed real-only test set using precision, recall, F1, IoU and Dice.

### Immediate action order

```text
1. Manually audit 400 generated images.
2. Record cracked, non-cracked and unclear counts.
3. Improve and validate the automatic classifier.
4. Compare manual and automatic crack-presence rates.
5. Decide between keeping 64 x 64 or moving to 128 x 128.
6. Run the downstream utility experiment.
```

## Formal Evaluation Set

The formal comparison uses 1,000 individual generated images for each condition:

```text
outputs/evaluation/conditional_cracked/
outputs/evaluation/conditional_non_cracked/
```

Both sets will be compared against the same real validation images and evaluated with the same FID, KID and LPIPS implementation used for the unconditional baseline.

## Conditional Evaluation Results

### Dataset verification

```text
Conditional cracked images:     1,000
Conditional non-cracked images: 1,000
Real validation comparison:     1,000
Image resolution:                  64 x 64
Device:                            CUDA
```

### Cracked condition metrics

```text
FID:                    0.701002
KID mean:               0.556873
KID standard deviation: 0.202802
Paired LPIPS mean:      0.425783
```

### Non-cracked condition metrics

```text
FID:                    0.724640
KID mean:               0.756606
KID standard deviation: 0.389974
Paired LPIPS mean:      0.416435
```

### Comparison with unconditional baseline

```text
                         FID       KID mean    LPIPS
Unconditional baseline   5.604571  3.419779    0.548679
Conditional cracked      0.701002  0.556873    0.425783
Conditional non-cracked  0.724640  0.756606    0.416435
```

### Interpretation

Under the same TorchMetrics configuration, both conditional distributions are substantially closer to the real validation distribution than the unconditional baseline. The cracked condition has slightly lower FID and KID than the non-cracked condition.

These metrics measure overall image-distribution similarity, not crack correctness. The lower conditional scores do not by themselves prove that the cracked condition contains visible cracks. Crack-presence rate must still be measured manually or with a separately validated crack classifier.

## Loss Capture

`src/plot_training_loss.py` compares the unconditional and conditional mean noise-prediction MSE histories from their CSV logs. The generated figure is saved at:

```text
outputs/plots/training_loss_comparison.png
```

## Limitations to Track

## Measured Training Performance

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Image resolution: 64 x 64
Training images sampled per epoch: 50,482
Batch size: 32
CUDA mixed precision: enabled
Training duration for 50 epochs: approximately 1 h 53 min
Conditional U-Net parameters: 3,258,627
```

The conditional model adds 512 trainable parameters for the two-class embedding compared with the 3,258,115-parameter unconditional U-Net.

- A binary crack label does not describe crack geometry.
- SDNET2018 is patch-level and does not cover every structural element.
- A generated image can look cracked without being physically meaningful.
- Image-quality metrics alone cannot establish engineering validity.
