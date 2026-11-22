# Experiment 5 Assumptions and Implementation Record

Revision: 2026-11-22. This current ledger supersedes unsupported interpretations in the exact archived pre-revision ledger. The full research report provides the methods, results, uncertainty and source evidence.

## A1 Canonical frame

Implementation or rationale: The lower image edge represents a tension-side reference after curation.

Interpretation and validation status: A synthetic image convention; cropped field images lack independently verified member/load frames.

## A2 Augmentation

Implementation or rationale: Horizontal flipping preserves the convention; vertical flipping does not.

Interpretation and validation status: Implemented augmentation choice, not proof of physical orientation.

## A3 Curvature

Implementation or rationale: Piecewise mask orientation differentiates straight and curved patterns.

Interpretation and validation status: Sensitive to segmentation, component selection and crop extent.

## A4 Taper

Implementation or rationale: EDT widths summarize apparent lower/upper width.

Interpretation and validation status: Pixel-scale proxy; no calibrated opening displacement or strain measurement.

## A5 Multiplicity

Implementation or rationale: Parallel diagonal components approximate torsion-like appearance.

Interpretation and validation status: A single face does not identify a member-wide torsional mechanism.

## A6 Alignment

Implementation or rationale: Horizontal splitting approximates reinforcement-parallel corrosion appearance.

Interpretation and validation status: Actual bar layout, corrosion and causation are unobserved.

## A7 Dispersion

Implementation or rationale: Multi-component masks approximate shrinkage-map appearance.

Interpretation and validation status: Texture, other damage processes and detector fragmentation are alternative explanations.

## Evaluation rules and observed consequences

The evaluator uses local-darkness mask extraction (23-pixel window, threshold 24.5 and 2 by 2 opening), then component geometry. Flexural acceptance requires detected presence and a near-vertical principal orientation; flexure-shear requires presence and piecewise angular change of at least 14 degrees; web-shear and splitting use diagonal/horizontal orientation predicates. Torsion accepts at least two diagonal or significant components, and map acceptance uses at least three components and concentration below 0.65, without the same presence gate. These predicates explicitly depend on the requested class. Historical torch-fidelity FID/KID selects up to 16 training-reference patches per class and uses eight KID subsets of size 12. It is a training-reference diagnostic, not an unseen-data validation.

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


## Errors identified in earlier interpretations

| Earlier issue | Current interpretation |
| --- | --- |
| Mechanism acceptance described as validated physical accuracy | Report requested-class predicate acceptance and identify the absence of expert mechanism labels. |
| Presence and acceptance percentages combined as one quantity | Retain separate denominators and predicate definitions, especially for torsion and map classes. |
| Training references described as independent validation | The historical fidelity evaluator selects reference patches from the training manifest. |
| RC box geometry assumed reliable throughout | The later qualified view removes 481 affected base records; historical models are not retroactively corrected. |

## Evidence and version history

Read [the full research report](EXPERIMENT_5_FULL_REPORT.md), [input signatures](report_evidence.json), and [archive index](report_archive/20261122-original/archive_index.json). Original statements are retained verbatim in the archive, including the full prior engineering rationale; they are historical assertions, not validated physical laws inferred from these images.
