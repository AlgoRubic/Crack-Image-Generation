# Trained checkpoint bundles

The source repository intentionally does not store model weights in normal Git history. The checkpoint files are large binary artifacts, so they are distributed separately as ZIP bundles.

## What is included

- `checkpoint-bundle-experiments-crack_type_conditional-checkpoints-*.zip` — Experiment 4 conditional crack model checkpoints
- `checkpoint-bundle-experiments-structural_mechanism_ddpm-checkpoints-*.zip` — Experiment 5 structural-mechanism checkpoints
- `checkpoint-bundle-experiments-exp6_level_c_spatial_physics-checkpoints-*.zip` — Experiment 6 spatial-physics checkpoints

Each numbered part is a separate ZIP archive. Download every part for an experiment, then extract each archive into the repository root while preserving paths. The archives contain paths beginning with `experiments/`, so a normal extraction into the project directory restores the expected layout.

The website, reports, metrics, result figures, and source code are already included in this repository. Raw datasets and the Python virtual environment remain local-only and are not redistributed.
