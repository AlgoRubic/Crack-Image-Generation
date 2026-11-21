# Experiment Results, Error Analyses & Full Reports (`experiments/`)

This folder holds the complete experimental outputs, training loss logs, generated test datasets, sample-by-sample error logs (`PER_SAMPLE_ERROR_LOG.csv`), statistical evaluation summaries (`EVALUATION_METRICS.json`), 4-panel error plots (`exp<N>_error_analysis_plots.png`), and full research reports (`EXPERIMENT_<N>_FULL_REPORT.md`) for all 6 experiments:

| Subfolder | Experiment | Generated Test Set | Error Analysis Status |
| :--- | :--- | :--- | :--- |
| `unconditional_ddpm/` | **Experiment 1** ($64 \times 64$ Unconditional) | `1,000` images | **Complete** (`EVALUATION_METRICS.json`, `PER_SAMPLE_ERROR_LOG.csv`, `exp1_error_analysis_plots.png`) |
| `conditional_ddpm/` | **Experiment 2** ($64 \times 64$ Binary Conditional) | `2,000` images (`1,000`/class) | **Complete** (`EVALUATION_METRICS.json`, `PER_SAMPLE_ERROR_LOG.csv`, `exp2_error_analysis_plots.png`) |
| `conditional_ddpm_128_augmented/` | **Experiment 3** ($128 \times 128$ Augmented Conditional) | `400` images (`200`/class) | **Complete** (`EVALUATION_METRICS.json`, `PER_SAMPLE_ERROR_LOG.csv`, `exp3_error_analysis_plots.png`) |
| `crack_type_conditional/` | **Experiment 4 (Level A)** ($256 \times 256$ 5-Class Morphology) | `80` images (`16`/class) | `EVALUATION_METRICS.json`, `FID_KID_METRICS.json`, `EXPERIMENT_4_FULL_REPORT.md` |
| `structural_mechanism_ddpm/` | **Experiment 5 (Level B)** ($256 \times 256$ 7-Class Mechanism) | `112` images (`16`/class) | `EVALUATION_METRICS.json`, `EXPERIMENT_5_FULL_REPORT.md` |
| `exp6_level_c_spatial_physics/` | **Experiment 6 (Level C)** ($256 \times 256$ Spatial-Physics) | **`1,400` images (`200`/class)** | Corrected evaluation: **87.86% morphology-heuristic agreement**, not validated mechanism accuracy. See `evaluation_corrected/`; older records are archival. |
