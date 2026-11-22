# Reproducing this project

Completed check (22 November 2026): two independent invocations analysed all 4,992 saved images, matched the numeric baseline observations for every experiment, and produced identical metric JSON in the same local environment. The verified run names are `saved-analysis-20261122-v2` and `saved-analysis-20261122-v3`; evidence is in `experiments/saved_analysis_reproduction.json`. To verify these local records and their retained inputs again, run `python pipeline/verify_reproductions.py --run-ids saved-analysis-20261122-v2 saved-analysis-20261122-v3`.

There are three distinct operations: checking the published evidence, reanalysing saved images, and generating new images from trained models. Full new research training is a separate operation. A successful evidence check does not establish independent scientific validity.

## 1. Check report evidence

From the repository root:

```powershell
python pipeline/verify_research_reports.py
python website/verify_site.py
```

The first command checks report sections, artifact signatures, archived originals and available saved confusion counts. It requires the committed evidence files; it does not require loading a model or training data. Text signatures permit only explicitly recorded Git newline normalization. Archived original bytes are checked strictly.

## 2. Reanalyse the existing evaluation images

Use the existing local environment, or install Python 3.10 and the research dependencies described in the root README. Run from the project root:

```powershell
.\.venv\Scripts\python.exe pipeline/reproduce_results.py --run-id my-review-001
```

Select experiments when needed:

```powershell
.\.venv\Scripts\python.exe pipeline/reproduce_results.py --run-id exp6-review-001 --experiments 6
```

The command checks that the full historical image sets are present: Exp1 1,000; Exp2 2,000; Exp3 400; Exp4 80; Exp5 112; Exp6 1,400. Experiments 1–3 also need their historical processed training CSVs and the deterministic reference subset used by the original morphology analysis. Exp6 requires the paired `phys_*.npy` commanded masks. Missing inputs fail before any evaluation starts. Exp4–6 do not need raw training data for this morphology-only rerun.

Each experiment writes to `experiments/<name>/reproductions/<run-id>/`. Existing run names are refused. Each run contains:

- Fresh `EVALUATION_METRICS.json` and `PER_SAMPLE_ERROR_LOG.csv` (including new per-image records for Exp4/5).
- Diagnostic plots where the evaluator implements them, plus a captured evaluation log.
- An exact copy of the baseline metrics for comparison.
- `run_manifest.json`: every image/reference/mask hash, evaluator hash, environment versions, output hashes, numeric comparison results and original-input preservation checks.

All numeric observations except fidelity and the manually authored Exp4 baseline-comparison block are compared against the saved baseline with absolute tolerance 0.0001 and zero relative tolerance. Missing numeric fields count as differences. A run returns a nonzero exit status on analysis failure, altered inputs or numeric differences; inspect `run_manifest.json` and `evaluation.log`. It never alters a baseline to force agreement. Run twice under different names to check repeatability on the same environment. Plot byte identity across different operating systems or library versions is not promised.

This operation uses CPU analysis and never invokes training or sampling. Fidelity is deliberately recorded as `not_requested`: earlier FID/KID runs used different feature extractors, sample budgets and reference preprocessing. Exact historical fidelity reproduction cannot be claimed without matching those protocols and references. For a new controlled held-out fidelity comparison, use the documented `pipeline/fidelity_research.py` workflow in the root README.

The raw evaluation images and commanded masks are local artifacts excluded from ordinary Git. Cloning the repository alone therefore supports report verification and pretrained generation, but does not provide all inputs for this exact saved-image reanalysis. Preserve these folders together with the run manifests when handing the local project to a collaborator. Obtain any required real datasets from their original sources according to their terms; the repository does not redistribute them.

## 3. Generate new images without retraining

```powershell
python pipeline/download_models.py
python pipeline/generate_cracks.py --experiment 6 --count 2 --seed 2026 --output-dir experiments/exp6_level_c_spatial_physics/my-generation
```

Use `python pipeline/generate_cracks.py --help` for class selection and sampling settings. The installer verifies published weight hashes. The inference program records its command and checkpoint; it never trains. The fresh CPU-install evidence in `experiments/fresh_install_cpu_verification.json` records a completed one-image inference check for every experiment using publicly downloaded weights.

New seeded images are not automatically byte-identical to historical samples whose complete generation settings were not recorded. Reproducing the old saved-image statistics and measuring a newly generated population are different experiments. Device, library versions, batch partition and sampler settings can affect generated pixels; keep each generation manifest with its outputs.

## 4. Repeat prospective research training

See the root README, `pipeline/research_protocol.json` and `experiments/clean_campaign_plan.json`. These specify versioned v5 splits, geometry-qualified data for Exp5/6, saved random states, EMA validation and matched controls. The full clean multi-seed campaign and independent expert validation remain pending. Historical released checkpoints cannot be relabelled as trained on the new clean splits.

## File preservation and portability

Do not move `data/` files or edit CSV path columns casually. Repository-relative paths in new manifests use forward slashes for portability. Published final weights are immutable. New analyses use new run directories. Keep caches and downloads inside `D:\Downloads\concrete_ddpm` on the original Windows machine; the saved-image reproduction command sets its temporary/cache locations there automatically.
