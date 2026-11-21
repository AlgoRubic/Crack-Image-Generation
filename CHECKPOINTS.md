# Trained checkpoint bundles

The preferred inference package is `crack-generation-final-models-v1.zip`: six final pretrained weights with the paths expected by `pipeline/generate_cracks.py`, plus `models/checkpoint_manifest.json`. Extract it into the repository root and run `python pipeline/checkpoints.py verify`. It does not require datasets or retraining. These are historical trained models, not new research_v4-clean models.

The compact bundle is available from the [historical model release](https://github.com/AlgoRubic/Crack-Image-Generation/releases/tag/historical-models-v1). It is 519,677,075 bytes; SHA256 is `b6c05b48d3faf0b5b4298a92f9ad8daf0bb31b1f65f489a5f7593031b7dc775a`. GitHub confirmed that digest, and the public download returned HTTP 200. Run `python pipeline/download_models.py` for automatic download and verification. It refuses to overwrite a different existing model. The older local multi-part bundles below retain full training history for Experiments 4–6 and are not needed for ordinary inference.

The source repository intentionally does not store model weights in normal Git history. The checkpoint files are large binary artifacts, so they are distributed separately as ZIP bundles.

## What is included

- `checkpoint-bundle-experiments-crack_type_conditional-checkpoints-*.zip` — Experiment 4 conditional crack model checkpoints
- `checkpoint-bundle-experiments-structural_mechanism_ddpm-checkpoints-*.zip` — Experiment 5 structural-mechanism checkpoints
- `checkpoint-bundle-experiments-exp6_level_c_spatial_physics-checkpoints-*.zip` — Experiment 6 spatial-physics checkpoints

Each numbered part is a separate ZIP archive. Download every part for an experiment, then extract each archive into the repository root while preserving paths. The archives contain paths beginning with `experiments/`, so a normal extraction into the project directory restores the expected layout.

The website, reports, metrics, result figures, and source code are already included in this repository. Raw datasets and the Python virtual environment remain local-only and are not redistributed.
