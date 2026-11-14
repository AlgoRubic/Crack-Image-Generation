# Experiment 5 (Level B) Execution Plan: Structural Failure Mechanism Conditioned DDPM at Native 256×256

**Project:** Synthetic Concrete-Crack Image Generation Research (`D:\Downloads\concrete_ddpm`)  
**Experiment Number:** Experiment 5 (Level B)  
**Experiment Directory:** `D:\Downloads\concrete_ddpm\experiments\structural_mechanism_ddpm`  
**Base Checkpoint for Transfer Learning:** `experiments/crack_type_conditional/checkpoints/crack_type_256_epoch_050.pt` (13.82M parameters)  
**Status:** Proposed Plan — Awaiting User Approval Before Execution  

---

## 1. Objective: Teaching the Model Structural Failure Mechanisms

In **Experiment 4 (Level A)**, the model learned **pure geometry** (`longitudinal`, `transverse`, `diagonal`, `alligator_map`). However, in structural civil engineering, a crack is defined by the **internal stress mechanism** that caused it.

The goal of **Experiment 5 (Level B)** is to train our $256 \times 256$ `AttentionUNet256` model to distinguish, understand, and synthesize **7 distinct structural engineering states**:

| Class ID | Mechanism Label | Structural Mechanics Signature & Visual Characteristics |
| :---: | :--- | :--- |
| **0** | `non_cracked` | Undamaged reinforced concrete surface with intact aggregate and cement matrix. |
| **1** | `flexural` | **Bending Tension Crack:** Initiates perpendicular to the **bottom tension edge** ($\theta \approx 90^\circ$), widest at the bottom edge ($w_{\text{bottom}} > w_{\text{top}}$), tapering upward and terminating near the neutral axis. |
| **2** | `flexure_shear` | **Combined Bending + Shear Crack:** Starts vertically at the bottom tension face ($\theta_{\text{lower}} \approx 80^\circ\text{--}90^\circ$) and **curves/bends diagonally** ($\theta_{\text{upper}} \approx 35^\circ\text{--}55^\circ$) as it propagates upward into the web. |
| **3** | `web_shear` | **Diagonal Tension Crack:** Forms in the interior web region at $\theta \approx 30^\circ\text{--}60^\circ$ (or $120^\circ\text{--}150^\circ$), **widest near the mid-depth neutral axis** and tapering toward both ends. |
| **4** | `torsion` | **Helical / Echelon Diagonal Cracking:** Characterized by **multiple parallel diagonal cracks** ($\ge 2$ co-directional inclined fractures at $\sim 45^\circ$ spacing across the member face) with edge distress. |
| **5** | `corrosion_splitting` | **Rebar Bond / Cover Splitting Crack:** Continuous horizontal fracture ($\theta \approx 0^\circ$) aligned along the longitudinal reinforcement bar axis, maintaining uniform width across the span. |
| **6** | `shrinkage_map` | **Plastic / Drying Shrinkage Crazing:** Multi-directional interconnected polygonal network ($C_{\text{ratio}} < 0.50$, multiple Y-junctions) driven by isotropic surface volume contraction. |

---

## 2. Physics-Guided Structural Labeling & Curation Pipeline

Why couldn't Level A tell `flexural`, `flexure_shear`, `web_shear`, and `torsion` apart? Because Level A only computed a single global angle $\theta$ on random crops.

In **Experiment 5**, we build a **Structural Mechanics Feature Extractor** (`src/prepare_experiment5_mechanisms.py`) that analyzes full-resolution masks and member-aligned $256 \times 256$ windows using **5 physical descriptors**:

1. **Boundary Initiation & Termination Profile ($B_{\text{touch}}$):**
   * Checks whether the crack skeleton touches the bottom tension boundary ($y \in [240, 255]$), top boundary ($y \in [0, 15]$), left/right boundaries, or terminates internally near the neutral axis ($y \in [80, 180]$).
2. **Piecewise Skeleton Curvature ($\Delta \theta = |\theta_{\text{lower}} - \theta_{\text{upper}}|$):**
   * Splits the skeleton into lower (tension zone) and upper (web zone) halves.
   * If $\theta_{\text{lower}} \in [70^\circ, 110^\circ]$ (vertical) and $\theta_{\text{upper}} \in [25^\circ, 65^\circ]$ (diagonal) with $\Delta \theta \ge 18^\circ$, it is classified as **`flexure_shear`**.
3. **Aperture Width Taper Profile ($w(s)$ via Euclidean Distance Transform):**
   * Computes the local crack width $w(s) = 2 \cdot \text{EDT}(s)$ along the skeleton arc length $s$.
   * **Flexural:** Width is maximal near the bottom edge and tapers upward ($w_{\text{lower}} / w_{\text{upper}} \ge 1.25$) with vertical orientation.
   * **Web-Shear:** Diagonal orientation ($\theta \in [30^\circ, 60^\circ]$) with maximum width in the mid-depth interior ($w_{\text{mid}} \ge w_{\text{ends}}$).
4. **Parallel Echelon Multiplicity ($N_{\text{echelon}}$):**
   * Detects $\ge 2$ distinct diagonal connected components ($\ge 45\text{ px}$ each) sharing parallel principal angles ($|\theta_1 - \theta_2| \le 18^\circ$). This uniquely identifies **`torsion`** echelon cracking.
5. **Canonical Structural Orientation Alignment:**
   * Ensures every training patch is oriented in its **canonical structural frame** (tension edge at bottom for flexural/flexure-shear, longitudinal rebar axis horizontal for corrosion splitting).
   * **Important Change in Training Augmentation:** Unlike Experiment 4 (which used random vertical flips that turned bottom-edge flexural cracks upside down!), Experiment 5 **disables random vertical flips for gravity/member-oriented classes** so the model strictly learns that flexural cracks initiate at the bottom tension face and propagate upward.

---

## 3. Model Architecture & Transfer Learning Strategy

Instead of training from random initialization (which took ~8.5 hours for 50 epochs), we leverage **Transfer Learning** from our completed Experiment 4 checkpoint (`crack_type_256_epoch_050.pt`):

* **Backbone:** `AttentionUNet256` (**13.82 Million parameters**, `base_channels=40`, FlashAttention at $32 \times 32$ and $16 \times 16$).
* **Weight Initialization:**
  * Load 100% of the convolutional, residual, GroupNorm, timestep MLP, and FlashAttention weights from `crack_type_256_epoch_050.pt`.
  * Expand the `class_emb` table from 6 tokens (5 classes + null) to **8 tokens** (7 structural mechanism classes + Token 7 as the CFG unconditional null token), initializing common classes from their learned embeddings.
* **Spatial Coordinate Conditioning (Structural Position Awareness):**
  * During training, the model preserves top-to-bottom structural orientation (no vertical flipping on flexural/flexure-shear), allowing the U-Net padding and attention positional biases to anchor bottom-edge tension initiation vs. mid-web diagonal initiation.
* **Hardware & Memory Limits (Strictly Enforced):**
  * **Parameters:** **13.82 Million** (exceeds $\ge 12\text{M}$ requirement).
  * **GPU VRAM:** **~3.5 – 4.4 GB** at batch size 6 with FP16 AMP.
  * **System RAM:** **$\le 3.2\text{ GB}$**.
  * **Training Duration:** **25 Fine-Tuning Epochs** ($\sim 3.5\text{ to } 4\text{ hours}$ total instead of 8.5 hours, because concrete surface textures and crack continuity are already learned!).

---

## 4. Planned Execution Phases

### Phase 1: Physics-Guided Dataset Curation (`src/prepare_experiment5_mechanisms.py`)
* Extract and classify native $256 \times 256$ concrete patches from DeepCrack, CrackForest, RC_10519 (mask-segmented), and SDNET2018 into the **7 structural mechanism classes** using the 5 physical descriptors (boundary initiation, piecewise curvature $\Delta \theta$, EDT width taper $w(s)$, parallel echelon count, and junction topology).
* Generate a visual verification grid of real training examples per mechanism class to confirm physical accuracy before training.
* Save dataset manifest to `data/processed_exp5_mechanisms/train.csv`.

### Phase 2: Architecture Adaptation & Transfer Learning Setup (`src/train_exp5_mechanisms_256.py`)
* Adapt `AttentionUNet256` for `num_classes=7` (plus CFG null token).
* Load pre-trained backbone weights from `crack_type_256_epoch_050.pt`.
* Replace random vertical flip augmentation with structure-preserving augmentations (horizontal flip only, preserving tension-bottom / compression-top geometry and diagonal shear angles).

### Phase 3: Fine-Tuning Training (25 Epochs)
* Train with AdamW ($\text{lr}=1.0 \times 10^{-4}$), FP16 AMP, `WeightedRandomSampler`, and $15\%$ CFG dropout.
* Save checkpoints `exp5_mechanism_256_epoch_001.pt` through `epoch_025.pt` in `experiments/structural_mechanism_ddpm/checkpoints/`.

### Phase 4: 7-Class DDIM Sampling ($w=2.5$) & Structural Evaluation
* Generate $4 \times 4$ contact sheets (16 images per class, 112 images total) across all 7 mechanisms:
  * `non_cracked_256_grid.png`
  * `flexural_256_grid.png`
  * `flexure_shear_256_grid.png`
  * `web_shear_256_grid.png`
  * `torsion_256_grid.png`
  * `corrosion_splitting_256_grid.png`
  * `shrinkage_map_256_grid.png`
* Evaluate **Structural Mechanism Adherence** (bottom initiation rate for flexural, curvature bend $\Delta \theta$ for flexure-shear, mid-web taper for web-shear, multi-strand echelon rate for torsion) plus FID and KID.

### Phase 5: Complete 17-Section Experiment 5 Technical Report
* Compile `EXPERIMENT_5_FULL_REPORT.md` with full structural mechanics theory, stress tensor derivations ($\sigma_1, \sigma_3, \tau_{\max}$), and comparative benchmarks across Experiments 1–5.
