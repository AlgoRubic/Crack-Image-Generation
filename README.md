# Synthetic Reinforced-Concrete Crack Generation Framework (Experiments 1 to 6)

Browse the current [research archive and audit](https://concrete-crack-synthesis.beigetoad12.chatgpt.site/research.html) for all experiment plots, corrected evaluations, dataset audits and methodological limitations.

This repository implements six experiments in synthetic concrete-crack image generation, progressing from an unconditional 64 × 64 DDPM to spatial trajectory and aperture-field conditioning at 256 × 256. The corrected Experiment 6 evaluation gives **87.86% agreement with a morphology-only heuristic classifier on 1,400 generated images**. This is a descriptive proxy score, not independently validated structural-mechanism accuracy. The earlier 99.29% figure used target-dependent assignments and is superseded.

Research preparation: run `python pipeline/research.py prepare` to reconstruct original-image provenance, then `python pipeline/hash_source_images.py` for the resumable source-content audit. Run `python pipeline/prepare_acquisition_splits.py --version research_v4` to create the current prospective splits. This groups sibling SDNET tiles by inferred acquisition ID and connects identical byte and decoded-RGB content across all experiments. Earlier split versions remain audit history, not the training default. Creation refuses to overwrite existing version directories. Existing checkpoints have already seen the historical images, so these splits require a new training lineage. Raw image paths and historical training manifests are preserved. Near-duplicate and structure/site independence checks remain outstanding.

The new trainer is `pipeline/train_research.py` for Experiments 4–6. It starts a separate scratch lineage, uses fixed-seed validation noise, tracks EMA weights, and saves optimizer, AMP scaler and random states. `best.pt` is selected by EMA validation loss; use its `ema` key for inference. Historical generation commands continue to use the original pretrained models.

To run pretrained inference, create a Python 3.10 environment and install a matching PyTorch pair using the official wheel index, then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
# CPU alternative: use https://download.pytorch.org/whl/cpu for the PyTorch command.
```

Run `python pipeline/download_models.py` to download and checksum-verify all six historical models. Alternatively, download `crack-generation-final-models-v1.zip` from the [published model release](https://github.com/AlgoRubic/Crack-Image-Generation/releases/tag/historical-models-v1) and extract it into the repository root, preserving `models/weights/` paths. Verify with `python pipeline/checkpoints.py verify`, then use `pipeline/generate_cracks.py` with a fresh output directory. Raw datasets are not needed for inference. All six experiments passed one-image generation checks in an isolated source/weight copy without datasets, using the installed local environment. A fresh dependency installation has not yet been verified.

```powershell
# One training batch and one validation batch; only a pipeline check.
python pipeline/train_research.py --experiment 6 --run-id my-smoke-check --smoke

# Full research training must be launched deliberately in a fresh run directory.
python pipeline/train_research.py --experiment 4 --run-id exp4-clean-seed2026 --epochs 50 --batch-size 1 --seed 2026

# Resume a trusted local research checkpoint into a fresh run directory.
python pipeline/train_research.py --experiment 4 --run-id exp4-clean-seed2026-continued --epochs 60 --resume experiments/crack_type_conditional/research_runs/exp4-clean-seed2026/last.pt
```

Smoke runs have passed for Experiments 4–6. The two-epoch Experiment 6 restart check matched uninterrupted model and EMA tensors exactly on this machine. This is a bounded reproducibility check, not a cross-platform determinism guarantee. Full research training, expert label validation, external evaluation and ablations remain outstanding. See `pipeline/research_protocol.json` for the prospective protocol.

`pipeline/train_research.py --experiment 6 --conditioning class_only` provides a matched five-channel control with constant null spatial fields. The default Experiment 6 mode uses spatial fields; both modes share architecture, data and image preprocessing. A class-only control smoke run passed. The [prepared clean campaign](experiments/clean_campaign_plan.json) contains 12 explicit training commands and measured short-batch runtime estimates; it has not been launched. The [batch-6 GPU profile](experiments/training_resource_profile_batch6.json) measured about 2.9 GiB peak allocated memory. Runtime estimates vary substantially with laptop load and exclude checkpoint overhead; they are planning estimates, not observed full-training durations.

For a newly trained Experiment 4–6 research checkpoint, sample with `pipeline/sample_research.py` (EMA by default), then evaluate the saved artifacts with `pipeline/evaluate_research.py`. Replace `RUN_ID` with the training run directory:

```powershell
python pipeline/sample_research.py --checkpoint experiments/exp6_level_c_spatial_physics/research_runs/RUN_ID/best.pt --run-id evaluation-seed2026 --seed 2026 --count 200
python pipeline/evaluate_research.py --samples experiments/exp6_level_c_spatial_physics/research_samples/evaluation-seed2026
```

These commands refuse smoke checkpoints/samples unless `--allow-smoke` is explicitly supplied. Sampling has no hidden color transform, records image/command/noise hashes, and never overwrites a run directory. Evaluation checks artifact hashes and class membership. Experiment 4 produces a five-class morphology matrix; Experiments 5/6 add balanced trajectory precision/recall/F1 with conditional distances. These remain morphology proxies. `pipeline/verify_research_pair.py` checks matched class-only/spatial protocols. The handoff smoke checks are infrastructure tests, not evidence of trained-model quality.

`pipeline/fidelity_research.py` prepares a class-balanced reference selection and, with `--compute`, runs standard `torch_fidelity` FID/KID using CPU float32 features and mini-batches of 8. It verifies reference-manifest signatures, acquisition/content disjointness, and decoded-pixel overlap against training patches. The default is preflight only; smoke samples can never be scored. Choose the same per-class budget, reference role, and preprocessing for all compared runs:

```powershell
python pipeline/fidelity_research.py --samples experiments/exp6_level_c_spatial_physics/research_samples/evaluation-seed2026 --run-id fidelity-validation --reference-split validation --per-class 16 --preprocessing rgb --compute
```

This is a command example, not a completed fidelity experiment. Small sample counts limit FID interpretation. The script fails rather than silently reducing an unavailable per-class budget. Optional `common_chroma` applies the same frozen transform to both real and generated images; do not mix that protocol with RGB scores. KID subset standard deviation is not a confidence interval.

---

## 1. Separated Repository Architecture (`datasets/`, `pipeline/`, `models/`, `experiments/`)

The project is cleanly separated into **four primary domains**, and **every folder contains both a `README.md` (human & tester documentation) and an `AGENTS.md` (AI agent operational rules & invariants)**:

```text
D:\Downloads\concrete_ddpm\
├── datasets/                                      # [1] SEPARATED DATASETS & MANIFESTS
│   ├── exp1_exp2_sdnet_64/                        #     50,482 SDNET2018 64x64 images (train.csv)
│   ├── exp3_augmented_128/                        #     61,001 SDNET2018 + RC_10519 128x128 images
│   ├── exp4_level_a_morphology_256/               #     13,139 5-Class Morphological 256x256 images
│   └── exp5_exp6_level_bc_mechanisms_256/         #     15,440 7-Class Structural Mechanism 256x256 images
│
├── pipeline/                                      # [2] SEPARATED CODE PIPELINE (INFERENCE, EVAL & TRAIN)
│   ├── generate_cracks.py                         #     ONE-COMMAND TESTER (No Retraining Required!)
│   ├── core/                                      #     Shared diffusion, U-Net & physics solver modules
│   ├── exp1_unconditional_64/                     #     Experiment 1 pipeline scripts
│   ├── exp2_conditional_64/                       #     Experiment 2 pipeline scripts
│   ├── exp3_conditional_128/                      #     Experiment 3 pipeline scripts
│   ├── exp4_level_a_morphology_256/               #     Experiment 4 (Level A) pipeline scripts
│   ├── exp5_level_b_mechanisms_256/               #     Experiment 5 (Level B) pipeline scripts
│   └── exp6_level_c_spatial_physics_256/          #     Experiment 6 (Level C) pipeline scripts
│
├── models/                                        # [3] SEPARATED MODEL ARCHITECTURES & PRETRAINED WEIGHTS
│   ├── architectures/                             #     SmallUNet, AttentionUNet256, GaussianDiffusion
│   └── weights/                                   #     Pre-trained final .pt weights for Experiments 1-6
│       ├── exp1_unconditional_64/                 #     exp1_unconditional_64_epoch050.pt (37.4 MB)
│       ├── exp2_conditional_64/                   #     exp2_conditional_64_epoch050.pt (12.5 MB)
│       ├── exp3_conditional_128/                  #     exp3_conditional_128_epoch050.pt (12.5 MB)
│       ├── exp4_level_a_morphology_256/           #     exp4_level_a_morphology_256_epoch050.pt (158.4 MB)
│       ├── exp5_level_b_mechanisms_256/           #     exp5_level_b_mechanisms_256_epoch025.pt (158.4 MB)
│       └── exp6_level_c_spatial_physics_256/      #     exp6_level_c_spatial_physics_256_epoch015.pt (158.4 MB)
│
└── experiments/                                   # [4] EXPERIMENT RESULTS, ERROR LOGS & FULL REPORTS
    ├── unconditional_ddpm/                        #     Exp 1: 1,000 test images + error analysis + report
    ├── conditional_ddpm/                          #     Exp 2: 2,000 test images + error analysis
    ├── conditional_ddpm_128_augmented/            #     Exp 3: 400 test images + error analysis
    ├── crack_type_conditional/                    #     Exp 4: Level A results + report
    ├── structural_mechanism_ddpm/                 #     Exp 5: Level B results + report
    └── exp6_level_c_spatial_physics/              #     Exp 6: Level C 1,400 test images + error analysis + report
```

---

## 2. Quick-Start: On-Demand Testing & Image Generation (Zero Retraining Required)

To generate any specific crack type (`flexural`, `flexure_shear`, `web_shear`, `torsion`, `corrosion_splitting`, `shrinkage_map`, `non_cracked`, or `all`) for any number of images (`1` to `1234+`):

```powershell
cd D:\Downloads\concrete_ddpm

# Generate 1,234 new flexural crack images using Experiment 6 (Level C) or Experiment 5 (Level B)
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type flexure --count 1234 --seed 999

# Generate 16 fixed-seed examples across all 7 classes
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type all --count 16 --seed 2026
```
