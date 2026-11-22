# Shared agent handoff

Updated: 22 November 2026, Asia/Kolkata. Workspace: `D:\Downloads\concrete_ddpm`.

This document gives any incoming agent the project context and operating workflow. Read [CURRENT_STATE.md](CURRENT_STATE.md) for the dated status, then inspect the live working tree. Current user instructions take priority; this handoff is not authorization to perform a new training campaign, change access controls or overwrite final checkpoints.

## What the project does

The project investigates diffusion-based synthetic concrete-crack image generation through six experiments. It progresses from unconditional texture generation to binary crack conditioning, higher resolution, five morphology classes, seven mechanism-appearance proxy classes, and spatial trajectory/aperture conditioning. It includes pretrained inference, historical saved-image evaluations, new research data audits, separate clean-training infrastructure, scientific reports and a public website.

The released models are historical checkpoints. They are usable without retraining and are not models trained on the newer clean research splits. Appearance labels and image-derived metrics do not establish the physical cause of a real structural crack.

| Experiment | Canonical experiment directory | Resolution and conditioning | Saved images |
|---|---|---|---:|
| 1 | `experiments/unconditional_ddpm/` | 64 × 64; unconditional | 1,000 |
| 2 | `experiments/conditional_ddpm/` | 64 × 64; cracked/non-cracked | 2,000 |
| 3 | `experiments/conditional_ddpm_128_augmented/` | 128 × 128; augmented binary data | 400 |
| 4 | `experiments/crack_type_conditional/` | 256 × 256; five morphology classes | 80 |
| 5 | `experiments/structural_mechanism_ddpm/` | 256 × 256; seven proxy mechanism classes | 112 |
| 6 | `experiments/exp6_level_c_spatial_physics/` | 256 × 256; seven classes plus spatial fields | 1,400 |

## First steps in a new session

1. Read root `AGENTS.md`, this document, `CURRENT_STATE.md`, and the nearest directory-specific `AGENTS.md` instructions.
2. Run `git status --short` and `git log -3 --oneline`. Inspect existing edits before changing a shared file. A dirty tree is expected; do not reset or clean it.
3. Read [README.md](README.md), [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md), and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
4. Choose the relevant experiment's `EXPERIMENT_<N>_FULL_REPORT.md` and `report_evidence.json`. Read machine-readable metrics and source code when making a numerical or implementation claim.
5. Confirm the user's current task and work on that scope. Pending scientific work is documented below; it is not an instruction to start every pending run.

## Folder ownership and preservation

| Location | Responsibility and constraints |
|---|---|
| `pipeline/` | Supported command-line interfaces, audits, preparation, reproduction and prospective research workflows |
| `src/` | Original evaluators/trainers and shared implementation; keep modules and imports compatible |
| `models/architectures/` | Network implementation |
| `models/weights/` | Final historical pretrained weights; never replace without explicit user confirmation |
| `datasets/` | Dataset documentation, versioned manifests, source and split audits |
| `data/` | Local raw/processed images and historical paths used by CSVs; excluded from ordinary Git |
| `experiments/<name>/` | Experiment-specific original outputs, evaluations, reports and new run directories |
| `website/` | Repository website source, full inline report mirrors and curated assets |
| `archive/` | Retained legacy suite/tools and an organization inventory; not an active pipeline |

Do not rename raw image folders or rewrite CSV path columns as cosmetic cleanup. Do not alter `report_archive/20261122-original/` bytes. The organization inventory attests 61 retained legacy files. The original `src/generate_docs_and_agents.py` is preserved for compatibility but contains superseded assumptions and must not generate the current records.

All new result runs must use unique names inside their own experiment directory. Inference defaults to `experiments/<name>/inference/seed<seed>/`; an explicit fresh `--output-dir` is preferred for named work. Analysis defaults to `experiments/<name>/reproductions/<run-id>/`. Never overwrite historical metrics to make a comparison pass.

## Environment and resource policy

- Windows PowerShell; the working project environment is `.venv\Scripts\python.exe`, Python 3.10.
- Use project-local `.cache/` for Torch, Matplotlib, temporary and dependency/download artifacts. Keep all project downloads in this project folder on D:. Do not install a new environment in the user's C: home as a convenience.
- The laptop GPU is an RTX 4050 with 6 GB VRAM. At 256 × 256, use CUDA FP16 AMP and batches/chunks no larger than 8, aiming below 4 GiB. CPU-only analysis does not need AMP.
- Read `requirements-core.txt`, `requirements-research.txt` and the README before changing dependencies. CUDA and CPU Torch wheels need matching torchvision builds. Do not confuse a locally generated dependency lock with a committed, independently verified installation protocol.
- Inference requests load trained weights and never start training. Full research training belongs to a separately named scratch lineage.

## Common commands

Run these from the repository root in PowerShell. They are task-specific options, not a checklist to execute on every session.

```powershell
Set-Location 'D:\Downloads\concrete_ddpm'

# Verify the report source/archive evidence and website content.
.\.venv\Scripts\python.exe pipeline/verify_research_reports.py
.\.venv\Scripts\python.exe website/verify_site.py

# Repeat the six saved-image analyses in NEW folders; no training, sampling or fidelity.
.\.venv\Scripts\python.exe pipeline/reproduce_results.py --run-id my-review-001

# Verify the two previously completed local reproduction runs and retained inputs.
.\.venv\Scripts\python.exe pipeline/verify_reproductions.py --run-ids saved-analysis-20261122-v2 saved-analysis-20261122-v3

# Install/check published weights and generate a small fresh pretrained sample.
.\.venv\Scripts\python.exe pipeline/download_models.py
.\.venv\Scripts\python.exe pipeline/checkpoints.py verify
.\.venv\Scripts\python.exe pipeline/generate_cracks.py --experiment 6 --crack-type flexural --count 2 --seed 2026 --output-dir experiments/exp6_level_c_spatial_physics/inference/agent-check-001

# Bounded infrastructure checks; not a full training/evaluation campaign.
.\.venv\Scripts\python.exe pipeline/check_research.py
```

Saved-image reproduction needs the local evaluation images; Exp1–3 also need their historical processed reference CSVs and selected real images. Exp6 needs each paired physics mask. A Git clone contains report evidence and code, not all local evaluation images or raw datasets. Use a fresh run ID if any example folder already exists.

## Scientific interpretation that must remain intact

- Exp2's 58.45%, Exp3's 64.25% and Exp6's corrected 87.86% are agreement with image morphology heuristics, not independently validated engineering accuracy.
- Exp6's primary metrics are `experiments/exp6_level_c_spatial_physics/evaluation_corrected/EVALUATION_METRICS.json`. Its root metrics are legacy evidence. The earlier 99.29% target-dependent assignment score is superseded.
- Exp5 target-predicate acceptance is not a full independent seven-class classifier. Some predicates lack a crack-presence gate. Do not label acceptance as physical-mechanism correctness.
- Exp4 did not establish resolved fragmentation. The current continuity report corrects that earlier interpretation.
- Historical FID/KID scores used incompatible extractors, sample counts, references or transforms. Keep protocols and lineages separate; do not present them as a controlled leaderboard.
- `research_v5` groups inferred acquisition identities, exact aliases and conservative similarity links. Zero candidates under the declared hash screen is not proof of complete independence.
- Exp5/6 prospective runs use the v5 `strict_rc_geometry_v1` view. Geometry flags do not independently validate mechanism labels.
- Exp6 procedural fields are commanded spatial templates. They are not an equilibrium fracture solver or proof of physically valid crack propagation.
- Smoke checks, restart checks, audits and exploratory detector/ablation pilots are bounded evidence. Full clean multi-seed training, generated-domain detector validation, independent expert labels and external evaluation remain pending.

## Reports and website updates

The current reports were authored through `pipeline/build_research_reports.py`; changing only the generated Markdown will be overwritten by the next build. Inspect the builder, edit the substantive narrative or source-backed calculations, rebuild and run `pipeline/verify_research_reports.py`. This builder requires local checkpoints and data manifests, unlike the evidence verifier. Source changes can invalidate recorded reproduction hashes; run a new named reproduction and update its evidence when appropriate instead of silently changing an old run manifest.

The site generator is `website/build_research_site.py`:

```powershell
.\.venv\Scripts\python.exe website/build_research_site.py --project-root D:/Downloads/concrete_ddpm
.\.venv\Scripts\python.exe website/verify_site.py
```

The user wants public access without a login, full reports displayed inline, equations rendered correctly, and result images in the result section rather than at the top. They previously reverted per-section dropdown changes: do not reintroduce those without a new request. Preserve the existing arrow alignment and report layout unless the task requests a design change.

Public website: https://concrete-crack-synthesis.beigetoad12.chatgpt.site

GitHub: https://github.com/AlgoRubic/Crack-Image-Generation

The Sites source checkout is `D:\Downloads\crack_research_site_checkout`, project ID `appgprj_6af2e54c44308191bb91022ec01193bb`. The GitHub repo's `website/` folder and the Sites source repository are distinct publication targets. For Sites changes, follow the available Sites tool/skill workflow: get the existing site and audience, obtain a fresh short-lived credential, open its source before editing, validate/package, save the exact pushed commit and deploy that saved version. Never put tokens into a document, a shell argument or Git. Do not create a replacement Site or change the audience as a shortcut. Recheck the live state rather than trusting expired credentials or this dated deployment snapshot.

Browser visual review was unavailable during the last publication; static checks and native deployment confirmation passed. If browser tooling becomes available, review the layout through its documented interface and respect any URL/access restrictions.

## Several agents working in this folder

- Share this handoff and the current-state document with every agent. Do not assume a new assistant inherits this chat.
- Agree on disjoint file ownership before concurrent edits. Shared report builders, evidence summaries, root docs and website publication need a single coordinating owner at a time.
- Use distinct run IDs and output folders. Record experiment, seed, checkpoint hash, source version, dependency versions and settings with results.
- Stage only reviewed task files. Do not commit all untracked local samples, helper scripts or user data using a blanket add. Do not include private reviewer keys, credentials, raw datasets or checkpoint binaries in ordinary Git.
- Finish with changed files, checks actually executed, observed outcomes, any required limitation, and the next concrete task. Update `CURRENT_STATE.md` without erasing useful prior evidence.

## Claude Code permission prompts

For the user's own Claude Code terminal session, the documented startup command is:

```powershell
Set-Location 'D:\Downloads\concrete_ddpm'
claude --dangerously-skip-permissions
```

The equivalent is `claude --permission-mode bypassPermissions`. This skips Claude Code permission prompts; it does not grant Windows administrator rights, defeat managed policies or confine shell commands to this folder. Some actions still require confirmation. Anthropic recommends using bypass mode in isolated environments. This handoff does not change any permission setting or launch that mode automatically.

Official references, checked 22 November 2026: [CLI reference](https://code.claude.com/docs/en/cli-reference) and [permissions](https://code.claude.com/docs/en/permissions).

## Starter instruction to give an incoming agent

> Work in D:\Downloads\concrete_ddpm. First read AGENTS.md, AGENT_HANDOFF.md, CURRENT_STATE.md and the applicable subfolder instructions. Inspect the working tree and preserve all existing edits, historical results and final weights. Use the documented pipeline and new experiment-local run directories; keep all downloads/cache/temp files in this project folder. Distinguish historical reproduced observations from prospective research. Complete my next requested task, run checks appropriate to the change, then update CURRENT_STATE.md with verified outcomes and remaining limitations.
