# Current project state

Snapshot: 22 November 2026, Asia/Kolkata. Workspace: `D:\Downloads\concrete_ddpm`.

Read [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for the shared agent workflow, commands, scientific interpretation and preservation rules. This file is a dated status record. Inspect the actual files, Git status and live deployment before acting.

## Completed work

- All six full research reports were revised against implementation, checkpoint metadata, training logs, saved metrics, data audits and preserved original records. Together they contain 18,424 words, including methods, equations, procedures, result images, uncertainty, assumptions, errors, conclusions, citations and reproduction evidence.
- Earlier report/figure bytes are preserved in each experiment's `report_archive/20261122-original/`. `report_evidence.json` records source and archive signatures, with explicit text newline handling for Git portability.
- The old duplicate replication suite is separated into `archive/legacy_replication_suite/`; a retired generator copy is in `archive/legacy_tools/`. An inventory records 61 retained files, verified byte-for-byte. Raw data paths, final model locations and import compatibility were preserved.
- A supported `pipeline/reproduce_results.py` command reruns saved-image morphology/spatial analysis in new experiment-local directories without training or replacing baselines. Exp4/5 now also have per-image CSVs in those new run directories.
- Two completed runs, `saved-analysis-20261122-v2` and `saved-analysis-20261122-v3`, each analysed all 4,992 saved images. All compared numeric baseline fields matched; the two fresh metric JSON records were identical in the same local environment. Original baselines and hashed inputs were preserved.
- A source-only export of commit `eca53ab` passed all six report evidence checks and all eight website content checks without copying the local raw data or pretrained weights. This establishes that the committed report verification inputs are present.
- Published-weight inference was previously checked for all six experiments in an isolated source/weight copy and a fresh Windows/Python 3.10 CPU environment. The fresh-install record reports `fresh_install_verified: true`; it is inference evidence, not proof of model quality.

## Historical experiment status

| Experiment | Saved image count | Current interpretation | Primary metric location within its experiment |
|---|---:|---|---|
| 1 | 1,000 | Detector flags 77.1% as cracked; unconditional prevalence, not accuracy | `EVALUATION_METRICS.json` |
| 2 | 2,000 | Binary heuristic requested-class agreement 58.45% | `EVALUATION_METRICS.json` |
| 3 | 400 | Binary heuristic requested-class agreement 64.25% | `EVALUATION_METRICS.json` |
| 4 | 80 | Five-class morphology diagnostics; fragmentation improvement not established | `EVALUATION_METRICS.json` |
| 5 | 112 | Seven target-predicate acceptance summaries; no independently validated mechanism accuracy | `EVALUATION_METRICS.json` |
| 6 | 1,400 | Corrected independent-of-target morphology assignment: 1,230/1,400, 87.86%; torsion recall 39% | `evaluation_corrected/EVALUATION_METRICS.json` |

Exp6's earlier 99.29% target-dependent assignment result is superseded. Its root `EVALUATION_METRICS.json` is a legacy record. The reproduction checks explicitly excluded historical fidelity values and the manually authored Exp4 baseline comparison. Do not describe all historical FID/KID scores as reproduced or comparable.

## Verified evidence and checks

| Evidence | Record |
|---|---|
| Two saved-analysis runs, all six experiments | `experiments/saved_analysis_reproduction.json` |
| Per-run inputs, environment, comparison and outputs | `experiments/<name>/reproductions/<run-id>/run_manifest.json` |
| Six report revisions and source checks | `experiments/report_revision_audit.json`, `experiments/report_verification.json` |
| 37 unit tests and 15 bounded check programs passed | `experiments/research_quality_checks.json` |
| Fresh CPU inference install | `experiments/fresh_install_cpu_verification.json` |
| Published checkpoint hashes | `models/checkpoint_manifest.json`, `models/release_manifest.json` |
| Retained legacy-file inventory | `archive/organization_manifest_20261122.json` |

These checks were completed during the report/reproduction work. Do not interpret this dated table as a new test run by a future agent. Browser visual review was not performed because the browser tooling was unavailable during publication; static content checks and native deployment confirmation passed.

## Code, publication and local state

- GitHub: https://github.com/AlgoRubic/Crack-Image-Generation ; branch `main`.
- Scientific/report baseline pushed to GitHub: `eca53ab` — `Organize legacy files and reproduce six saved-image analyses`. Prior report revision: `554cf15`; prior provenance audit: `e57fd72`. Handoff-only documentation commits may follow this baseline.
- Public website: https://concrete-crack-synthesis.beigetoad12.chatgpt.site . No project login was added.
- Latest confirmed deployment at this snapshot: Sites version **9**, status **succeeded**. Source commit: `98115821afc23bf23d9cf18fa65e1b31f72fdcdc` in the separate Sites repository.
- Sites project: `appgprj_6af2e54c44308191bb91022ec01193bb`; local checkout: `D:\Downloads\crack_research_site_checkout`. Never reuse credentials from a prior session; obtain a fresh one through the supported tools.
- Full current reports are shown inline on each experiment page. Result grids are in result sections; equations use MathJax. The repository's `website/` folder and the hosted source must both be updated when changing public content.

Existing modified tracked files observed before this handoff were:

```text
experiments/exp6_level_c_spatial_physics/EVALUATION_METRICS.json
experiments/exp6_level_c_spatial_physics/results/commanded_physics_metadata.json
```

Those pre-existing changes were deliberately not committed, reverted or replaced by the report work. Inspect them with the user/task context before any future staging. There are also local untracked instructions, historical images/plots, earlier reproduction runs, smoke-run artifacts, CPU helper scripts and a CPU dependency lock. Some are useful local records; not all have been reviewed for publication. Never blanket-stage, delete or treat them as failed runs merely because they are untracked. The current handoff creates additional documentation files; check live status for the latest set.

## Prospective research, not yet completed

The newer clean-training lineage uses `research_v5` splits and the `strict_rc_geometry_v1` view for Exp5/6. The protocol is `pipeline/research_protocol.json`, with a prepared campaign in `experiments/clean_campaign_plan.json`. The existing released weights remain historical and exposed to earlier data.

Still pending:

1. Full clean training and multi-seed evaluation under the declared prospective protocol.
2. Independent expert review of mechanism-appearance proxy labels and generated examples.
3. Generated-domain detector validation; real-image mask validation does not establish generated mechanism accuracy.
4. Matched class-only/spatial controls in the completed clean lineage, controlled held-out fidelity and external evaluation.
5. Visual browser review of the published report layout when supported browser access is available.

Existing audits, detector calibration and paired spatial pilots are exploratory or bounded evidence; their assumptions are described in the reports. The previous report improvement goal is complete, and the current request is to prepare a handoff/current-state record. No new full training campaign was started as part of those requests.

## Next agent's immediate actions

Read the handoff and applicable instructions, inspect the live working tree, then follow the user's next concrete task. If asked to reproduce local saved-image results, use a new run ID with `pipeline/reproduce_results.py`. If asked to improve reports, edit the authored builder, rebuild the reports/site and verify source evidence. Do not silently promote a pending scientific task to completed status.

When handing over again, update this snapshot with the actual source commit, edited files, commands executed, verified outcomes, deployment state and remaining work. Keep local paths and links portable where possible, and never include credentials or private reviewer keys.
