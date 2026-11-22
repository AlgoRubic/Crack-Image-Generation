# Project map

Start with the root README for the current scientific conclusions. The six experiment reports are the authoritative interpretation of historical results; old reports remain preserved as historical evidence.

| Folder | Purpose | What to use |
|---|---|---|
| `pipeline/` | Supported command-line entry points | `generate_cracks.py` for pretrained inference; `reproduce_results.py` for saved-image reanalysis; `train_research.py` for explicitly requested new research training |
| `src/` | Shared implementation and original experiment programs | Retained for import compatibility; prefer the supported pipeline commands |
| `models/architectures/` | Network definitions | Model source code |
| `models/weights/exp1..exp6/` | Published historical pretrained checkpoints | Immutable inference inputs; install with `pipeline/download_models.py` |
| `datasets/` | Versioned split manifests, acquisition documentation and data audits | `research_v5` and `strict_rc_geometry_v1` are prospective clean-training inputs |
| `data/` | Local raw/processed image files and historical manifests | Existing paths are retained because dataset CSVs refer to them; excluded from Git |
| `experiments/<experiment>/` | Reports, results, plots and run records | See the standard layout below |
| `website/` | Website source, curated figures and report mirrors | Public presentation of the research records |
| `docs/` | Navigation and reproduction instructions | Read `REPRODUCIBILITY.md` before repeating an analysis |
| `archive/` | Separated legacy replication suite and retired tool references | Historical reference only; not the active execution interface |
| `.venv/`, `.cache/` | Local Python environment and caches | Local only; all project-created temporary/cache files stay inside this project |

The older root `outputs/`, `checkpoints/` and `presentation_output/` paths remain for compatibility with existing scripts and saved artifacts. They are not alternative canonical locations for new research runs. No raw image directories, final weights or existing experiment outputs were renamed.

## Experiment directory convention

```text
experiments/<experiment>/
  EXPERIMENT_<N>_FULL_REPORT.md       current research report
  report_evidence.json              report input and archive signatures
  EVALUATION_METRICS.json            historical stored evaluation
  PER_SAMPLE_ERROR_LOG.csv          original per-image record, where available
  plots/                            current publication figures
  outputs/evaluation/               Exp1–3 saved evaluation images
  generated_samples/                Exp4–5 saved evaluation images
  results/                          Exp6 images and paired commanded masks
  evaluation_corrected/             Exp6 corrected primary evaluation
  reproductions/<run-id>/            NEW saved-image reanalysis, never overwrites originals
  research_runs/<run-id>/            separate clean-training lineage, where present
  report_archive/20261122-original/  exact pre-revision report and figure evidence
```

Not every historical experiment contains every optional directory. Exp6's root metrics are a legacy record; use `evaluation_corrected/EVALUATION_METRICS.json` for its corrected class-assignment results. A reproduction run records explicitly which baseline it compares against.

## Experiment names

| Number | Directory | Historical model |
|---|---|---|
| 1 | `unconditional_ddpm` | Unconditional 64 × 64 |
| 2 | `conditional_ddpm` | Binary class-conditional 64 × 64 |
| 3 | `conditional_ddpm_128_augmented` | Augmented binary 128 × 128 |
| 4 | `crack_type_conditional` | Five morphology classes, 256 × 256 |
| 5 | `structural_mechanism_ddpm` | Seven proxy mechanism classes, 256 × 256 |
| 6 | `exp6_level_c_spatial_physics` | Seven classes with spatial fields, 256 × 256 |

`archive/organization_manifest_20261122.json` records the source path, retained path, size and SHA-256 of every separated legacy file. The original `src/generate_docs_and_agents.py` remains in place for compatibility; its archived copy is reference-only and should not be used to regenerate the current reports.
