# Experiment 6 Assumptions and Implementation Record

Revision: 2026-11-22. This current ledger supersedes unsupported interpretations in the exact archived pre-revision ledger. The full research report provides the methods, results, uncertainty and source evidence.

## C1 Spatial fields

Implementation or rationale: Image-derived ridge and EDT fields encode visible crack geometry.

Interpretation and validation status: Self-supervised image proxies, not measured stress, strain or calibrated crack opening.

## C2 Command generator

Implementation or rationale: Procedural class templates approximate mechanism-inspired crack shapes.

Interpretation and validation status: The implementation samples shapes, angles, widths and warped cell boundaries; it is not a solved boundary-value fracture model.

## C3 Input-channel surgery

Implementation or rationale: Zero-initialized extra channels preserve the initial RGB convolution contribution.

Interpretation and validation status: Architectural initialization property; no guarantee of convergence, compliance or realism after training.

## C4 Chromatic transform

Implementation or rationale: Strong desaturation and channel offsets reduce unwanted tints.

Interpretation and validation status: Changes image distribution; clipping/quantization prevent a guarantee of perfect luminance preservation.

## C5 Joint CFG

Implementation or rationale: Class and spatial fields are dropped together during training.

Interpretation and validation status: Zero-field plus present-class ablations can be outside the historical conditioning distribution.

## Train/inference correspondence

Implementation or rationale: Image-extracted fields and procedural inference fields have compatible distributions.

Interpretation and validation status: Unverified domain-transfer assumption requiring controlled validation.

## Pixel geometry

Implementation or rationale: Distances and widths in pixels are meaningful image measurements.

Interpretation and validation status: No millimetre scale, material properties, equilibrium residual or structural safety assessment is established.

## Evaluation rules and observed consequences

The corrected evaluator predicts each class from image morphology alone, independent of the requested class and commanded field. Its fixed local-darkness mask uses window 23, threshold 24.5 and 2 by 2 opening. A second, requested-class-dependent acceptance predicate remains a separate descriptive quantity.

Command core pixels are $S_0>0.22$. Historical trajectory recall measures the fraction of command-core pixels covered after four dilation iterations of the detected mask. Distance statistics pool both command-to-detection and detection-to-command nearest-pixel distances; the stored 95th percentile is that pooled percentile, not a demonstrated physical error bound. The legacy evaluator substitutes 50 pixels for missing-detection MAE/RMSE and 80 pixels for its distance percentile. Such penalties must be identified rather than treated as measured distances. The later paired-field audit instead reports missing counts explicitly and leaves conditional distance undefined when no detection exists.

No new FID/KID score was requested in the corrected 1,400-image run (`fidelity_status: not_requested`). Earlier fidelity claims from different report versions must not be substituted into that run. New research fidelity code requires explicit reference selection and shared preprocessing, verifies manifest and pixel overlap, and refuses scientific scoring of smoke outputs.

## Completed evidence and remaining validation

The prospective **research_v5** manifests contain the following base partition counts. They were constructed after the historical training and do not retroactively make its checkpoints unseen-data models.

| Partition | Base records |
| --- | --- |
| train | 12,357 |
| validation | 1,565 |
| test | 1,518 |

The shared source audit covers 67,264 originals, with 65,866 distinct decoded RGB images and 1,398 extra exact decoded copies. The v4 perceptual screen found 386 cross-split candidates. V5 conservatively links all 1,111 candidate pairs, including within-split pairs, before assignment; the same pHash ≤3 and dHash ≤6 search then finds zero cross-split candidates. This limited, uncalibrated screen can miss crops, rotations, flips and larger edits. Source names and zero screened matches do not prove site or structure independence.

RC attribution identifies a matching Mendeley record [R9] and its requested paper [R10]. Local ZIP sizes/CRCs match extracted files, but no publisher digest established local archive identity. All 10,519 local JSON files contain crack rectangles; these boxes do not validate image-level labels, pixel masks or structural mechanisms. The geometry audit flags 1,145 source annotations, including 1,122 out-of-bounds cases and 150 image/annotation dimension mismatches; categories overlap.

The prospective `strict_rc_geometry_v1` view excludes 481 affected Exp5/6 records (371 train, 67 validation, 43 test), retains all seven proxy classes and preserves v5 component assignments. Exclusion is conservative; it does not prove every excluded label wrong or every retained label correct.

| Partition | Qualified records |
| --- | --- |
| train | 11,986 |
| validation | 1,498 |
| test | 1,475 |

Scratch training, EMA selection on fixed-noise validation, provenance-checked generation and geometry/fidelity preflight are implemented for the new lineage. Their bounded smoke runs are infrastructure checks. Full clean multi-seed training, independently labeled generated images, external validation and downstream detector utility have not been established by the historical artifacts. A prepared campaign is not a completed experiment.

Computational sharing was tested in a fresh Windows/Python 3.10 CPU environment: the public weights were downloaded and verified and one image per experiment was generated without raw datasets. This confirms that inference can execute under that environment; it does not reproduce model training or validate image quality.

### Detector validation and exploratory calibration

The fixed detector was compared with pixel masks on 51 DeepCrack originals (102 nonoverlapping native tiles). Its pixel precision/recall/IoU/Dice were 33.69% / 58.81% / 27.26% / 42.84%. A 24-setting validation-only search on 46 originals selected window 31, threshold 30 and opening 2. On the archived 51-original test selection, precision/recall/IoU/Dice became 42.36% / 61.25% / 33.41% / 50.08%. The original-image cluster-bootstrap paired IoU gain was 6.15 percentage points (95% interval 5.04–7.31). These selections belong to archived research_v1, and test results had previously been inspected, so this is exploratory detector evidence, not a new untouched test of the generator or generated-image mechanism labels.

### Paired field sensitivity pilot

The pilot contains 126 images: seven classes, three seeds, two commands per class/seed and three field conditions. Cracked-image summaries use 36 images per condition. Initial random states were matched and non-cracked controls were byte-identical. Outputs are evaluated against the original correct command.

| Detector | Input field | Cracked images | Trajectory recall | Missing detections | Conditional MAE px |
| --- | --- | --- | --- | --- | --- |
| fixed | correct | 36 | 98.61% | 0 | 0.474 |
| fixed | zero | 36 | 0.00% | 36 | Undefined |
| fixed | shuffled | 36 | 11.25% | 0 | 27.955 |
| calibrated | correct | 36 | 98.07% | 0 | 0.510 |
| calibrated | zero | 36 | 0.00% | 36 | Undefined |
| calibrated | shuffled | 36 | 11.07% | 0 | 27.918 |

This supports input-field sensitivity for the historical checkpoint. The zero-field/present-class combination may be out of distribution because historical dropout removed both together. It is not a separately trained matched class-only baseline and cannot establish physical causation. Distances must be read alongside missing-detection counts.


## Errors identified in earlier interpretations

| Earlier issue | Current interpretation |
| --- | --- |
| 99.29% labeled independent seven-class accuracy | Superseded by the corrected morphology-only matrix: 1,230/1,400 agreements. |
| Procedural generator described as guaranteed physically valid | The code does not solve a calibrated fracture boundary-value problem or validate equilibrium/material failure. |
| High recall treated as sufficient template fidelity | Report false positives, precision and missing detections alongside recall; distances are conditional when detections are absent. |
| Zero-field pilot described as a trained class-only baseline | It is an input ablation on a historical checkpoint; a separately trained matched class-only control remains pending. |
| New split and quality audits presented as validation of old models | The historical weights and transfer ancestors remain exposed to earlier data. |

## Evidence and version history

Read [the full research report](EXPERIMENT_6_FULL_REPORT.md), [input signatures](report_evidence.json), and [archive index](report_archive/20261122-original/archive_index.json). Original statements are retained verbatim in the archive, including the full prior engineering rationale; they are historical assertions, not validated physical laws inferred from these images.
