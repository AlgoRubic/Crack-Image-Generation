# Experiment 3: 128x128 Conditional DDPM with Additional Cracked Data

## Identity

Experiment name: 128x128 Conditional DDPM with Cracked-Image Augmentation

Experiment directory: `D:\Downloads\concrete_ddpm\experiments\conditional_ddpm_128_augmented\`

Project root: `D:\Downloads\concrete_ddpm\`

Status (updated 2026-11-06): Training complete. Experiment CLOSED by user decision on 2026-11-06: large-scale generation and formal FID/KID/LPIPS/crack-presence evaluation were NOT performed. No final Experiment 3 quality metrics exist.

> Note: the "Recommended New Data Sources" / "Planned" sections below are the original plan. The source actually used was RC_10519 (cracked subset only); DamSegment and Historical-Crack18-19 were not used.

## Objective

Train the conditional DDPM at 128 x 128 resolution using:

1. the existing SDNET2018 concrete images; and
2. a genuinely new external dataset containing additional concrete crack images.

The model should generate separate cracked and non-cracked concrete images while preserving more crack detail than the earlier 64 x 64 experiment.

## Important Dataset Decision

The 800-image Kaggle concrete crack segmentation dataset is not suitable as the primary new source because its data card states that it was selected from SDNET2018. Adding it to SDNET2018 would create duplication and possible train/test leakage.

Reference: https://www.kaggle.com/datasets/jakubniemiec/concrete-crack-images

## Recommended New Data Sources

### Primary candidate: DamSegment

DamSegment contains concrete-dam images with crack and spalling annotations, including classification, detection and pixel-level segmentation tasks. Use only images whose annotation identifies a crack for the additional cracked-image subset. Exclude spalling-only images unless explicitly needed.

Source: https://data.mendeley.com/datasets/z5z6gtt5t4/1

### Secondary candidate: Reinforced Concrete Structure Crack Detection Dataset 10519

This dataset contains 10,519 cracked reinforced-concrete images and an equal number of non-cracked images. Use only its cracked subset for the requested additional crack data, while retaining the existing SDNET2018 non-cracked data.

Source: https://data.mendeley.com/datasets/3ghxck2wd5/1

### Optional secondary source: Historical-Crack18-19

This contains approximately 757 cracked and 3,139 intact 256 x 256 wall patches derived from historical-building photographs. It is useful for weathered wall texture, but many patches originate from only about 40 raw photographs, so it must be split by source photograph to avoid leakage.

Source: https://data.mendeley.com/datasets/xfk99kpmj9/1

## Planned Dataset Composition

Initial recommended composition:

```text
Existing SDNET2018:
    cracked + non-cracked

New external data:
    cracked only
```

The final metadata must preserve:

```text
source_dataset
surface_type
crack_status
mask_available
original_record_or_photo_id
split
```

## Planned Preprocessing

1. Keep all original downloads under `data/raw/`.
2. Verify licenses and record source citations.
3. Remove exact and near duplicates against SDNET2018.
4. Convert images to RGB.
5. Resize to 128 x 128 using high-quality interpolation.
6. Normalize to [-1, 1].
7. Preserve crack masks separately if available.
8. Split by original photograph or specimen where possible.
9. Use cracked-only external images only in the training split.
10. Keep a real-only validation/test set that is not augmented with external images.

## Planned Label Policy

```text
class 0 = non_cracked
class 1 = cracked
```

Do not label a sample as cracked if it only contains spalling, stains, texture or uncertain damage unless the source annotation explicitly identifies a crack.

## Planned Model Changes

- Reuse the conditional U-Net architecture.
- Change image size from 64 x 64 to 128 x 128.
- Keep the 1,000-step diffusion schedule initially.
- Keep class conditioning for cracked/non-cracked output.
- Start with batch size 8 or 16 depending on GPU memory.
- Enable CUDA mixed precision.

## Planned Comparisons

Compare:

```text
Experiment 2:
    conditional DDPM, 64 x 64, SDNET2018

Experiment 3:
    conditional DDPM, 128 x 128, SDNET2018 + external cracked data
```

Use the same real-only test set and the same evaluation code.

## Planned Metrics

- training MSE loss;
- FID;
- KID;
- LPIPS;
- crack-presence rate;
- visual crack clarity;
- diversity and nearest-real similarity;
- downstream crack-classification or segmentation utility.

## Known Risks

- External datasets may contain overlapping images or derived crops.
- Cracked-only augmentation may cause the generated distribution to overproduce cracks.
- 128 x 128 training will require more memory and time.
- Pixel-level masks do not necessarily encode crack cause or structural severity.
- A lower FID does not prove better crack generation.

## Execution Status

Completed dataset preparation and training.

Combined training set:

```text
SDNET2018 non-cracked:        42,846
SDNET2018 cracked:             7,636
External cracked (RC_10519):  10,519
Total:                        61,001
```

Training configuration:

```text
Resolution: 128 x 128
Model: conditional SmallUNet/DDPM
Batch size: 16
Epochs: 50
Diffusion steps: 1,000
Device: CUDA, NVIDIA GeForce RTX 4050 Laptop GPU
Final training loss: 0.014057591692068844
```

Checkpoint:

```text
checkpoints\conditional_128_augmented\conditional_128_augmented_epoch_050.pt
```

Reference grids generated for visual inspection:

```text
outputs\reference_128\cracked_epoch_050.png
outputs\reference_128\non_cracked_epoch_050.png
```

Detailed PDF report:

```text
reports\experiment_3_conditional_128_augmented_report.pdf
```

The reference grids contain 16 images per class and show concrete-like textures with visible crack patterns in several cracked samples. A large-scale cracked generation run was intentionally stopped after 32 images so the project could be reorganized; those 32 partial images are retained at:

```text
outputs\evaluation\conditional_128_augmented_cracked\
```

They are not a complete evaluation set and must not be used for final FID/KID/LPIPS reporting.

Large-scale generation and FID/KID/LPIPS evaluation are pending. Do not report final quality metrics until those runs are completed.

Do not use the Kaggle 800-image SDNET-derived set as the additional source.

## Download and Extraction Record

Downloaded archive:

```text
D:\Downloads\Reinforced concrete structure crack detection data.zip
```

Preserved raw copy and extracted contents:

```text
data\raw\external_cracks\Reinforced concrete structure crack detection data\
```

Observed contents:

```text
images\       10,519 JPG images
label_json\   10,519 JSON annotation files (re-counted 2026-11-06; an earlier note said 5,512)
```

The JSON files use LabelMe format and contain a shape with the label `crack`. Some image files do not have a corresponding JSON annotation, so the integration script must preserve an `annotation_available` field rather than assume every image has a mask or box.

Observed sample dimensions include 227 x 227 and 256 x 256. All images will be converted to RGB and resized to 128 x 128 for this experiment.


## 2026-11-06 Verification and Fixes (closing record)

Verified (read-only):

- Final checkpoint `checkpoints\conditional_128_augmented_epoch_050.pt` (relative to this experiment folder; STATUS previously said `checkpoints\conditional_128_augmented\...`) loads strictly into `SmallUNet(num_classes=2)`, 3,258,627 params, all weights finite, epoch index 49 (= epoch 50), loss 0.014057591692068844.
- Training log: `logs\conditional_128_augmented_training_loss.csv` (not `outputs\logs\`).
- `data\processed_128\train.csv`: 61,001 rows = SDNET2018 non_cracked 42,846 + SDNET2018 cracked 7,636 + RC_10519 cracked 10,519.
- Zero overlap of original source paths between SDNET2018 validation (5,610) and Experiment 3 training.
- The 32 partial cracked images are exactly reproducible: regenerating with seed 2026, batch size 32 gave pixel-identical images (mean abs diff 0.0). They are batch 0 of the seed rule `seed = 2026 + batch_index`.
- Sampling speed: 32 images x 1,000 steps = 231 s on RTX 4050 Laptop (~7.2 s/image, 1.85 GB peak).

Fixes applied (originals kept as `src\*.bak_2026-11-06`, this file as `STATUS.md.bak_2026-11-06`):

- `src\generate_eval_128.py`: default output is now this experiment's `outputs\evaluation\conditional_128_augmented_<class>` (was the empty root `outputs\`); added `--output-dir`; deterministic batch-level resume that skips complete batches and never overwrites existing files; writes `manifest.json` and refuses to resume if class/batch size/seed differ. Tested: skip of existing batch 0 works, 32 files unchanged.
- `src\generate_reference_128.py`: output path now this experiment's `outputs\reference_128`.
- `src\train_conditional_128_augmented.py`: paths now point to this experiment folder and the script refuses to run if the completed epoch-50 checkpoint exists (prevents accidental overwrite).

Known evaluation-protocol limitations (documented, not changed, relevant to Experiments 1-2 metrics):

- `src\evaluate_metrics.py` resizes all images to 64 x 64 and uses Inception `feature=64` (not the standard 2048); values are only comparable within this project.
- Its real reference is the first 1,000 sorted validation PNGs (858 non_cracked / 142 cracked), not class-matched.
- `checkpoints\crack_classifier.pt` does not exist, so `src\measure_crack_presence.py` cannot run.

If evaluation is ever resumed:

```powershell
cd D:\Downloads\concrete_ddpm; .\.venv\Scripts\Activate.ps1
python .\src\generate_eval_128.py --checkpoint .\experiments\conditional_ddpm_128_augmented\checkpoints\conditional_128_augmented_epoch_050.pt --class-name cracked --count 1000 --batch-size 32 --seed 2026
```
