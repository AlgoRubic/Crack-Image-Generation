# DDPM Concrete Crack Experiment Notes

This is a living record of decisions, suggestions, limitations, errors, observations and improvements made during the experiment.

## Experiment Scope

The first experiment trains an unconditional DDPM on concrete images containing both cracked and non-cracked surfaces from SDNET2018.

The first objective is to test whether the DDPM can learn realistic concrete surface texture and crack-like visual patterns. It does not yet control the structural element or explicitly request a cracked/non-cracked output.

## Dataset Decision

### Decision

Use both cracked and non-cracked SDNET2018 images.

### Reason

The model should learn the broader distribution of concrete surfaces, including healthy texture and damaged texture. This is more useful for a later concrete-condition generation system than training only on cracked images.

### Limitation

Because the first model is unconditional, it cannot be instructed to generate a cracked image or a non-cracked image on demand. The generated proportions follow the learned training distribution.

### Future improvement

Use a conditional DDPM with a crack-status label and, later, structural-element labels such as bridge deck, wall, beam or column.

## Dataset Split Decision

### Decision

Use 90% of the images for training and 10% for validation, stratified by surface type and crack label.

### Reason

This keeps the validation set representative of bridge decks, pavements, walls, cracked images and non-cracked images.

### Limitation

SDNET2018 consists mostly of 256 x 256 image patches. Multiple patches may originate from related original photographs, so random image-level splitting can still produce related samples in both splits.

### Future improvement

If original-image identifiers become available, split by original photograph rather than by individual patch.

## Image Resolution Decision

### Decision

Preprocess the first experiment at 64 x 64 pixels.

### Reasons

- It reduces GPU memory usage.
- It makes the first DDPM experiment faster.
- It allows the complete training and sampling pipeline to be tested affordably.
- It is appropriate for a small baseline experiment.

### Limitation

Very thin or hairline cracks may disappear or become blurred when 256 x 256 images are reduced to 64 x 64. The model may learn broad concrete texture without preserving the finest crack details.

### Alternative: 128 x 128

Use 128 x 128 if the 64 x 64 samples do not preserve visible crack structures.

Advantages:

- Better preservation of thin and branching cracks.
- More spatial detail for the U-Net.
- More useful images for later crack analysis.

Disadvantages:

- Higher GPU memory usage.
- Slower training and sampling.
- May require a smaller batch size.
- May require a larger U-Net or more training time.

### Planned comparison

First train and verify the 64 x 64 baseline. If cracks are not visibly learned, repeat the experiment at 128 x 128 and compare generated samples and metrics.

## Image Format Decision

### Decision

Convert images to RGB and save processed copies as PNG.

### Reason

RGB gives every image the same three-channel input format. PNG avoids adding another lossy JPEG compression step to already compressed source images.

### Limitation

PNG files require more storage than JPEG files.

## Raw Data Preservation Decision

### Decision

Keep the original dataset unchanged in `data/raw/` and write processed copies to `data/processed/`.

### Reason

This preserves the original data and allows the experiment to be repeated at another resolution or with different preprocessing.

### Limitation

The project requires additional disk space because both raw and processed images are retained.

## Current Processing Result

Processed image counts:

```text
Training images:    50,482
Validation images:   5,610
Skipped images:          0
Image size:             64 x 64
Channels:                    3 RGB
```

## Diffusion Schedule Phase

### Decision

Use a 1,000-step linear beta schedule for the first DDPM baseline.

### Reason

This is a standard, understandable starting point. The forward process gradually adds Gaussian noise, allowing the model to learn the reverse denoising process later.

### Implementation

The schedule is implemented in `src/diffusion.py`. The forward equation is:

```text
x_t = sqrt(alpha_bar_t) * x_0
      + sqrt(1 - alpha_bar_t) * epsilon
```

### Limitation

A linear schedule may not be optimal for preserving very fine crack details. If the baseline performs poorly, a cosine schedule can be tested later.

### Verification

`src/test_diffusion.py` checks that a clean image becomes progressively noisier at timesteps 0, 100, 300, 600, 900 and 999.

## U-Net Phase

### Decision

Use a compact U-Net with residual blocks, sinusoidal timestep embeddings, GroupNorm, skip connections and a base channel width of 64.

### Reason

The U-Net is the noise-prediction network used by the DDPM. Skip connections help preserve fine spatial information while the downsampling path learns broader concrete texture and the upsampling path reconstructs image-sized noise predictions.

### Input and output

```text
Input:  noisy RGB image [batch, 3, 64, 64] plus timestep
Output: predicted noise [batch, 3, 64, 64]
```

### Limitation

This small model is designed for a fast baseline. It may not have enough capacity to represent very fine crack patterns at higher resolution. A larger U-Net or attention blocks may be required for 128 x 128 images.

### Verification

`src/test_model.py` checks that the U-Net runs on the available device, produces the same shape as its input and returns finite values.

## Training Loop Phase

### Decision

Train the U-Net to predict the Gaussian noise added by the forward diffusion process using mean squared error loss.

### Training sequence

```text
clean image -> random timestep -> noisy image -> U-Net prediction
            -> compare predicted and actual noise -> update weights
```

### Initial settings

```text
Batch size: 64, adjustable for GPU memory
Learning rate: 2e-4
Optimizer: AdamW
Gradient clipping: maximum norm 1.0
Checkpoint: after each epoch
Loss: mean squared error
```

### Limitation

The first implementation is unconditional. The crack and surface labels are preserved in metadata but are not supplied to the U-Net. Therefore, the model cannot be directed to generate a particular surface type or crack status yet.

### Verification

`src/test_training.py` performs one complete forward pass, loss calculation, backward pass and optimizer update on a batch of four images. Full training is intentionally separate because it will use the GPU for a longer period.

## Training Speed Optimization

### Decision

Use CUDA automatic mixed precision and configurable data-loader workers.

### Reason

Mixed precision reduces the memory and computation cost of compatible operations on the RTX 4050. Multiple data-loader workers can prepare image batches while the GPU is training the previous batch.

### Implementation

`src/train.py` now uses:

- CUDA autocasting when a CUDA device is available.
- `GradScaler` for stable mixed-precision backpropagation.
- `pin_memory` on CUDA.
- Configurable `--num-workers`.
- Persistent workers when `--num-workers` is greater than zero.

### Limitation

The best number of workers depends on the Windows system, storage device and available RAM. Start with two workers. If loading becomes unstable, use `--num-workers 0`.

### Resume decision

Epoch 1 was completed before this optimization. Resume from `ddpm_epoch_001.pt` and run 49 additional epochs to reach a total of 50 epochs.

### Expected runtime

The resumed run uses:

```text
Remaining epochs: 49
Batch size: 32
Data-loader workers: 2
CUDA mixed precision: enabled
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

An approximate expected duration is 3 to 8 hours for the remaining 49 epochs. The actual time depends on GPU temperature, disk speed, Windows background activity and data-loader performance. The first completed optimized epoch provides the best estimate for the remaining run.

### Runtime measurement

After one optimized epoch, estimate the remaining time as:

```text
remaining time = optimized epoch duration x 48
```

This estimate should be recorded after the first resumed epoch.

## Training Completion

### Result

The 50-epoch DDPM training run completed successfully. Checkpoint `ddpm_epoch_050.pt` is available in `checkpoints/`.

### Final logged loss

The final logged training loss was approximately `0.0160`. Loss alone does not prove that the generated images are realistic, so reverse sampling and visual inspection are required.

## Reverse Sampling Phase

### Objective

Generate new images by starting from Gaussian noise and repeatedly applying the learned denoising model from timestep 999 down to timestep 0.

### Implementation

`src/sample.py` loads a trained checkpoint and saves a grid of generated 64 x 64 images in `outputs/samples/`.

### Limitation

Sampling with 1,000 reverse steps can take several minutes. The first sample grid may contain blurry or incomplete crack structures. This is a baseline result and must be evaluated visually before deciding whether more training or higher resolution is required.

### First sample observation

The epoch-50 sample grid was generated successfully at:

```text
outputs/samples/samples_epoch_050_seed_42.png
```

The samples show concrete-like color, aggregate and surface texture. Crack visibility is inconsistent: some images contain faint or partial crack-like structures, while others mainly show texture. This suggests that 64 x 64 resolution may be limiting fine crack preservation, although additional seeds and quantitative checks are still needed before changing the resolution.

### Proposed next comparison

Generate additional sample grids with different random seeds. Then compare the 64 x 64 baseline against a 128 x 128 experiment or a higher-capacity model if crack structures remain weak.

## Crack Visibility Diagnosis

### Observation

The generated samples do not show clearly visible cracks in a reliable proportion of images. They primarily reproduce concrete texture and aggregate appearance.

### Likely causes

- The model is unconditional, so it is not instructed to generate a crack.
- Non-cracked images greatly outnumber cracked images in SDNET2018.
- Resizing from 256 x 256 to 64 x 64 can remove hairline cracks.
- A small U-Net has limited capacity for fine crack geometry.
- A low MSE noise-prediction loss does not by itself guarantee visually obvious cracks.

### Decision

Treat the current mixed-data unconditional DDPM as a valid texture baseline, but do not claim that it is a reliable crack generator.

### Recommended next experiment

Train a conditional DDPM using the existing crack-status labels. Provide the U-Net with a class embedding:

```text
class 0 = non-cracked
class 1 = cracked
```

Generate two separate sets and compare them:

```text
noise + cracked label     -> cracked concrete images
noise + non-cracked label -> healthy concrete images
```

### Resolution recommendation

Before moving directly to 128 x 128, test the conditional approach at 64 x 64. This isolates the effect of conditioning. If cracks remain unclear, then repeat the conditional experiment at 128 x 128.

### Alternative quick baseline

A crack-only DDPM can be trained as a diagnostic experiment, but it cannot generate non-cracked images. It is useful for checking whether the model can learn crack appearance at all, not as the final controlled system.

## Evaluation Status and Metrics

### What has been established

The current experiment has established that:

- The complete training pipeline runs successfully on the RTX 4050.
- The DDPM can generate new 64 x 64 concrete-like images from random noise.
- The generated images reproduce broad concrete texture, aggregate appearance and color variation.
- Crack generation is not reliable or clearly controllable in the unconditional mixed-data model.

### What has not been established

The current experiment has not yet proven that:

- the generated images are statistically similar to real images;
- the generated images contain the correct proportion of cracks;
- the generated images improve a crack detector or segmentation model;
- the model has learned all concrete structural crack types;
- the model avoids memorizing training images.

### Training loss

The model uses mean squared error between actual and predicted noise:

```text
MSE(actual noise, predicted noise)
```

The final logged loss was approximately `0.0160`.

This confirms that the denoising objective is being optimized, but a low noise-prediction loss does not guarantee realistic or clearly cracked images.

### Visual inspection

Generated grids should be inspected for concrete texture, visible cracks, crack continuity, branching, width variation, diversity, blur, repeated patterns and artifacts. This is necessary but subjective.

### FID and KID

FID compares feature distributions of real and generated images; lower values generally indicate closer distributions. KID is another distribution metric that is less biased for smaller sample sizes.

Limitation: generic Inception features are not specialized for concrete cracks. Good FID or KID may reflect color and texture while missing thin crack geometry.

### LPIPS and diversity

LPIPS can measure perceptual similarity. Use it to check both similarity to real images and diversity among generated images. Excessive similarity may indicate memorization or mode collapse.

### Crack-presence rate

Calculate:

```text
crack-presence rate = generated images classified as cracked / total generated images
```

Use a separately validated crack classifier or manual expert annotation for a small sample. Do not treat the original training labels as labels for generated images.

### Downstream utility

Train a crack detector or segmentation model in two settings:

```text
real training images only
real + synthetic training images
```

Evaluate both on the same real-only test set using precision, recall, F1-score, IoU and Dice. Improvement on unseen real images is stronger evidence than visual quality alone.

### Memorization check

Compare generated images with the nearest real training images using perceptual similarity. Very close matches may indicate memorization rather than useful generation.

### Current evaluation conclusion

```text
Training loss: measured
Visual quality: inspected
FID/KID: not yet measured
Crack-presence rate: not yet measured
Downstream detector improvement: not yet measured
  Memorization: not yet measured
  ```

### Evaluation implementation status

`src/evaluate_baseline.py` provides lightweight diagnostics using pixel statistics, generated-sample diversity and nearest-real-image distance. These are diagnostic proxies, not substitutes for standard FID, KID or LPIPS.

The specialized metric packages are not currently installed in the project environment. No packages were installed automatically. Standard FID/KID/LPIPS evaluation will be added only after selecting and approving a feature-extractor implementation.

## Baseline Evaluation Results

### Measured numbers

```text
Training images: 50,482
Validation images: 5,610
Training epochs: 50
Image resolution: 64 x 64
Model parameters: 3,258,115
Final logged training loss: 0.01599615436508108
Generated images in inspected grid: 16
Sampling checkpoint: ddpm_epoch_050.pt
Sampling seed: 42
```

### Metrics not yet numerically measured

```text
FID: not measured
KID: not measured
LPIPS: not measured
Crack-presence rate: not measured with a validated classifier
Downstream detection/segmentation improvement: not measured
Memorization similarity: not measured
```

The lightweight evaluation script was created and syntax-checked, but it was not used to claim formal FID, KID or LPIPS values. The current sample was stored as a contact-sheet grid rather than separate image files, so a formal sample-level evaluation must be performed after exporting individual generated images.

### Findings

- The DDPM training and sampling pipeline completed successfully on CUDA.
- The model generated concrete-like texture, aggregate appearance and color variation.
- Crack visibility was inconsistent; several samples primarily showed concrete texture rather than clear cracks.
- The unconditional mixed-data model cannot be instructed to generate a crack.
- The non-cracked majority, 64 x 64 resolution and loss-based training objective likely reduced the visibility of thin cracks.

### Baseline conclusion

The experiment is a successful proof of concept for unconditional concrete-image generation, but it is not yet a reliable controlled concrete-crack generator. The next model must add crack-status conditioning, and its evaluation must include separate cracked/non-cracked sample sets and a measured crack-presence rate.

### Evaluation tooling update

TorchMetrics 1.9.0 and Torch-Fidelity are now installed in the project environment. The project contains:

- `src/generate_eval_set.py` for saving individual generated images;
- `src/evaluate_metrics.py` for FID, KID and paired LPIPS diagnostics.

The same real validation subset, sample count, resolution and evaluation code must be used for the conditional model.

### Measured distribution metrics

The first formal evaluation used 1,000 generated images and 1,000 real validation images:

```text
Device: CUDA
Images evaluated: 1,000
FID: 5.604571
KID mean: 3.419779
KID standard deviation: 1.028021
Paired LPIPS mean: 0.548679
```

### Interpretation

- FID `5.604571` is the baseline distribution-similarity value. Lower is better when comparing the same evaluation protocol.
- KID mean `3.419779` is the baseline kernel-distribution distance. Lower is better; the standard deviation is `1.028021`.
- Paired LPIPS `0.548679` indicates perceptual distance between the generated images and the paired real images. It is not a standalone quality score because the generated and real images are not semantically matched pairs.

These metrics indicate that the generated set has measurable similarity to the real concrete-image distribution, but they do not prove that cracks are visible or structurally meaningful. The conditional model must use the same real set, sample count, resolution and metric implementation for a fair comparison.

## Measured Training Performance

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Image resolution: 64 x 64
Training images per epoch: 50,482
Batch size: 32
CUDA mixed precision: enabled for the resumed run
```

Measured from checkpoint timestamps:

```text
Unconditional DDPM: approximately 2 h 15 min for 50 epochs
Conditional DDPM:   approximately 1 h 53 min for 50 epochs
```

Parameter counts:

```text
Unconditional U-Net: 3,258,115 trainable parameters
Conditional U-Net:   3,258,627 trainable parameters
```

The conditional model adds 512 parameters for the two-class embedding. Its faster measured runtime is affected by run conditions and should not be interpreted as a general speed guarantee.

## Future Notes Template

For every new decision, record:

```text
## Topic

### Decision

### Reason

### Limitation

### Evidence or observation

### Future improvement
```
