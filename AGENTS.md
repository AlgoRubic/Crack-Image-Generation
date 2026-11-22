# AGENTS.md — Root Operational Guidelines (`D:\Downloads\concrete_ddpm`)

## Start here for every agent

Read [AGENT_HANDOFF.md](AGENT_HANDOFF.md), [CURRENT_STATE.md](CURRENT_STATE.md), and the `AGENTS.md` files applicable to the directories you will edit. The handoff is shared across Codex, Claude Code, Antigravity and other assistants; it does not override the current user's instructions. Check the actual working tree before acting because the state document is a dated snapshot.

- Keep project downloads, temporary files and dependency caches under `D:\Downloads\concrete_ddpm`; the user has prohibited project downloads outside the chosen folder.
- Preserve existing data, final weights, historical results and report archives. Write new evaluations into fresh experiment-local run directories.
- Use `pipeline/` entry points. Do not run `src/generate_docs_and_agents.py` or the archived replication suite to regenerate current reports.
- Inspect existing changes before editing or staging. Never discard another agent's or the user's work. Assign separate file ownership when several agents work at once; coordinate shared report builders and publication.
- Record completed changes, verification and remaining limitations in `CURRENT_STATE.md` when handing work over. Do not describe a prospective run or a smoke check as a completed scientific experiment.

## 1. Repository Separation Invariants
- **Datasets (`datasets/` & `data/`):** All training/validation manifests (`train.csv`) and dataset documentation reside in `datasets/`, while raw/processed image binaries remain in `data/` so legacy relative paths in CSV manifests never break.
- **Pipeline (`pipeline/` & `src/`):** All executable scripts are organized by experiment inside `pipeline/` (`pipeline/generate_cracks.py`, `pipeline/core/`, `pipeline/exp1..exp6/`). Never delete `src/` as existing scripts reference `src/` imports.
- **Models (`models/`):** Neural network definitions live in `models/architectures/` and final pre-trained weights (`.pt`) are isolated per experiment in `models/weights/exp1..exp6/`. Never overwrite final pre-trained checkpoints without explicit user confirmation.
- **Experiments (`experiments/`):** Contains generated evaluation outputs, `EVALUATION_METRICS.json`, `PER_SAMPLE_ERROR_LOG.csv`, `exp<N>_error_analysis_plots.png`, and `EXPERIMENT_<N>_FULL_REPORT.md`.

## 2. Hardware & Memory Constraints
- **Target GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (`6.0 GB VRAM`).
- Always use FP16 Automatic Mixed Precision (`torch.amp.autocast("cuda")`) and mini-batch chunking (`batch_size <= 8` at $256 \times 256$) so GPU VRAM usage stays below `4.0 GB`.
