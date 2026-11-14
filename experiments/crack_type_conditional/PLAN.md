# Experiment 4 (PLAN ONLY, not implemented): Crack-Type-Conditioned Generation

Status: planning, 2026-11-06. Nothing in this plan has been run yet.

## 1. The core problem to settle first

"Shear", "torsion", "flexural" are **mechanism** labels. They describe *why* a crack formed. You usually tell them apart by:

- the crack's orientation relative to the member axis;
- its position on the member (support region, midspan, all faces);
- the loading.

A 128 x 128 close-up patch (SDNET2018, RC_10519) shows only texture and a crack segment. A 45-degree crack in a patch could be shear, torsion, shrinkage or settlement; the patch alone cannot decide. Labelling patches as "shear" or "torsion" would give unverifiable labels and invalid engineering claims.

So the plan uses two label levels:

| Level | Classes | Visible in a patch? | Data needed |
|---|---|---|---|
| A. Morphology (geometry) | transverse/vertical, longitudinal/horizontal, diagonal (~30-60 degrees), map/alligator, branched, plus non_cracked | Yes | Patches + crack masks |
| B. Mechanism (structural) | flexural, shear (web-shear / flexure-shear), torsion (spiral, all faces), shrinkage/plastic, settlement, corrosion-induced (along rebar) | Only with element context | Whole-member images (beam/column/wall) with loading or location metadata, labelled by an engineer |

Separate non-crack damage classes stay distinct: spalling, efflorescence/stain, exposed rebar, uncertain. They must never be merged into "cracked".

Recommendation: build Level A first (achievable with data we partly have). Treat Level B as a separate, data-dependent phase.

## 2. Data

Already available:

- SDNET2018: cracked/non-cracked only, no masks, no type labels.
- RC_10519: 10,519 cracked images with 10,519 LabelMe JSON files. Audited 2026-11-06: all 16,157 shapes are `crack|rectangle`, i.e. **bounding boxes only, no pixel masks**.
  - Box aspect and placement give only a coarse orientation hint (a tall box suggests vertical, a wide box horizontal, a square box is ambiguous).
  - Pixel masks must come from a mask dataset (Crack500, DeepCrack, CFD, dacl10k), or from a crack segmenter trained on those and applied to RC_10519/SDNET, followed by manual QC.
  - Morphology is then derived from masks: skeletonise, principal-axis or Hough angle, branch count, area/length ratio for map cracking.

Candidates to evaluate (licence, availability and label quality to be verified before use):

- **CODEBRIM**: bridge defects (crack, spallation, efflorescence, exposed bars, corrosion stain). Useful for the non-crack damage classes.
- **dacl10k**: bridge-inspection semantic segmentation with many damage classes (including crack and alligator crack).
- **RDD2022**: road damage with longitudinal, transverse and alligator crack classes. These are pavement images and have a domain shift.
- **Crack500, DeepCrack, CFD**: pixel masks, no type labels. Usable for morphology derivation.
- **Level B**: public mechanism-labelled datasets are scarce. Options:
  - lab beam/column test photo sets from the literature;
  - manual engineer labelling of whole-element photos;
  - collecting our own.

  This is the main risk.

Rules carried over from Experiments 1-3:

- keep provenance per image;
- split by source photo/specimen;
- de-duplicate across datasets;
- keep real-only validation/test sets;
- do not use SDNET-derived subsets as "external".

## 3. Labelling pipeline (Level A)

1. Obtain masks:
   - use the masks that come with mask datasets;
   - for RC_10519/SDNET, use pseudo-masks from a real-data crack segmenter, restricted to the RC_10519 boxes, then manually QC a sample.
2. Automatic morphology tagging from masks:
   - dominant angle gives transverse / longitudinal / diagonal;
   - branch points plus coverage give map/alligator;
   - low-confidence cases go to `uncertain`.
3. Manually verify a stratified random sample (e.g. 300 images). Report agreement between the automatic tags and the human labels (Cohen's kappa). Fix the rules if agreement is low.
4. Drop or keep `uncertain` explicitly; never force a class.
5. Save the label table:
   `image, source, split, crack_status, morphology, angle_deg, mask_path, label_method (auto/manual), confidence`.

## 4. Model design

Lessons from Experiments 2-3: plain class embedding gave weak crack control (about 5/16 visibly cracked in Exp2), and evaluation was hard because generated images have no ground truth. Recommended design: **two-stage, mask-conditioned generation.**

```text
Stage 1  type label (+ optional angle/width)  ->  crack MASK generator
         (small DDPM on 128x128 binary masks, or sample/augment real masks)
Stage 2  crack MASK + type label  ->  concrete IMAGE DDPM
         (mask concatenated as extra input channel; type as embedding)
```

Why:

- The crack type becomes geometrically controllable and **checkable**: compare the conditioning mask with the generated crack.
- Every synthetic image comes with a pixel mask, which is directly usable for segmentation augmentation (the project's end goal).

Training details:

- **Classifier-free guidance**: drop labels/masks about 10% of the time in training and use guidance scale w in {1, 2, 3, 5} at sampling. This is the main fix for weak conditioning.
- Larger U-Net with attention at 16 x 16 and 32 x 32 (the current SmallUNet has 3.26 M params and no attention).
- 128 x 128 first; 256 x 256 only if crack width/detail is still lost.
- **DDIM sampling (50-100 steps)** to cut the current ~7.2 s/image to under 1 s.

Simpler fallback (Plan B): single-stage DDPM with a multi-class embedding (non_cracked + morphology classes) and CFG. Cheaper, but no masks and weaker verification.

## 5. Evaluation (decided before training)

1. **Type adherence**:
   - train an independent morphology classifier on *real* data only, report per-class precision/recall/F1 on a real held-out set;
   - apply it to generated images only if macro-F1 is acceptable;
   - plus a blinded manual audit of e.g. 50 images per class.
2. **Mask adherence (Stage 2)**: an independent crack segmenter (trained on real data) runs on the generated image; report IoU/Dice against the conditioning mask, and angle error.
3. **Image quality**:
   - FID/KID with standard 2048-dim features at native resolution;
   - against **class-matched** real references (fixes the Exp1-3 protocol weakness);
   - keep the old 64/feature-64 protocol only for backward comparison.
4. **Diversity / memorisation**: nearest-real neighbours in feature space, plus duplicate checks.
5. **Downstream utility**: train a segmentation (U-Net) or type classifier on real only vs real + QC-passed synthetic; test on the same real-only test set, 3 seeds.

No mechanism claim (shear/torsion) is made unless Level B data with engineer labels exists.

## 6. Phases

| Phase | Work | Rough cost |
|---|---|---|
| 0 | Decide taxonomy and scope (questions below) | - |
| 1 | Acquire a mask dataset; train a real-data crack segmenter; pseudo-mask RC_10519/SDNET cracked; auto morphology tagging; manual check of 300 | ~3-5 GPU h + ~1-2 days incl. review |
| 2 | Candidate dataset survey and licence check; optional download of CODEBRIM/dacl10k/Crack500 | ~1 day |
| 3 | Real-data morphology classifier and crack segmenter (evaluators) | ~2-3 GPU h |
| 4 | Stage 2 mask-to-image DDPM with CFG, 128 x 128 | ~10-20 GPU h (RTX 4050) |
| 5 | Stage 1 mask generator (or real-mask sampling first) | ~3-6 GPU h |
| 6 | Evaluation suite (section 5) | ~3-5 GPU h |
| 7 | Downstream segmentation test | ~3-6 GPU h |
| 8 (optional) | Level B mechanism labels with element-level data | depends on data |

Disk: about 5-15 GB (datasets plus checkpoints) on D:. Nothing goes on C:.

## 7. Decisions needed from the user

1. Accept the two-level taxonomy (morphology now, mechanism later)? Or is Level B (shear/torsion/flexural) a hard requirement now? If so, which source of element-level labelled images?
2. Go with the mask-conditioned two-stage design (recommended) or the simpler single-stage multi-class model?
3. May additional public datasets (CODEBRIM, dacl10k, Crack500, RDD2022) be downloaded, subject to licence check?
4. Who does the manual verification (step 3.3 and the audit in 5.1)?
