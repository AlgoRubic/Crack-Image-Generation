# Experiment 5 (Level B): Complete Engineering Documentation, Theoretical Assumptions, and Step-by-Step Rationale

**Experiment Name:** Structural Failure Mechanism Conditioned DDPM at Native $256 \times 256$  
**Experiment Number:** Experiment 5 (Level B)  
**Directory:** `D:\Downloads\concrete_ddpm\experiments\structural_mechanism_ddpm`  
**Author:** Antigravity Pairing Session  
**Date:** 2026-11-10  

---

## 1. Purpose of This Document

Per the user's strict documentation requirement:
> *"Document everything for the experiments: every step, the reason, every assumption, and all."*

This document serves as the comprehensive engineering ledger for **Experiment 5 (Level B)**. Every dataset transformation, physical mechanics formula, geometric proxy assumption, data augmentation restriction, transfer-learning weight projection, and hyperparameter choice is explicitly recorded and justified below.

---

## 2. Fundamental Engineering Problem & Core Assumptions

### 2.1 Why Experiment 4 (Level A) Was Not Sufficient for Structural Mechanisms
In Experiment 4, cracks were categorized strictly by **first-order 2D image geometry** (`longitudinal` $\approx 0^\circ$, `transverse` $\approx 90^\circ$, `diagonal` $\approx 45^\circ$, `alligator_map`).
* **The Limitation:** In structural mechanics, two cracks with the exact same average angle can originate from completely different internal stress states:
  1. A vertical crack ($\theta \approx 90^\circ$) initiating at the bottom edge of a beam and tapering upward is a **Flexural (bending tension) crack**.
  2. A crack that starts vertically at the bottom edge ($\theta_{\text{lower}} \approx 90^\circ$) and bends diagonally ($\theta_{\text{upper}} \approx 45^\circ$) into the web is a **Flexure-Shear crack** caused by combined bending moment $M$ and shear force $V$.
  3. A single straight diagonal crack ($\theta \approx 45^\circ$) widest in the mid-depth web is a **Web-Shear (diagonal tension) crack** where principal tensile stress $\sigma_1 = \frac{\sigma_x}{2} + \sqrt{\left(\frac{\sigma_x}{2}\right)^2 + \tau_{xy}^2}$ exceeds the tensile strength of concrete $f_{ct}$ in the uncracked web.
  4. Multiple parallel diagonal cracks ($\theta_1 \approx \theta_2 \approx 45^\circ$) spaced in echelon formation across the face are **Torsional cracks** resulting from circulatory shear flow $\tau = \frac{T}{2 A_o t}$ wrapping helically around the member.

### 2.2 Explicit Assumptions Made to Map $256 \times 256$ Patches to Structural Mechanisms
Because public crack datasets (DeepCrack, CrackForest, RC_10519, SDNET2018) do not come with laboratory load-cell metadata or finite-element strain gauges attached to each photo, we must formulate **explicit, verifiable structural-mechanics assumptions** that map observable crack morphology, curvature, boundary contact, width taper, and multi-crack multiplicity to the 7 structural failure mechanisms:

#### Assumption A1: Canonical Member Coordinate Frame
* **Statement:** We assume the horizontal axis ($x \in [0, 255]$) aligns with the longitudinal axis of the reinforced concrete member (beam/girder/wall), and the bottom boundary ($y = 255$) represents the extreme tensile fiber under positive bending moment.
* **Reason:** Without a fixed canonical coordinate frame, "bottom-tension initiation" has no meaning in a neural network's $256 \times 256$ pixel grid.
* **Enforcement:**
  1. During dataset curation (`src/prepare_experiment5_mechanisms.py`), if a vertical flexural crack touches the top edge ($y \le 20$) or is wider at the top than the bottom ($w_{\text{upper}} > 1.15\, w_{\text{lower}}$), the patch is canonically flipped vertically so that the tension root always sits at the bottom ($y \to 255$).
  2. Similarly, if a curved flexure-shear crack has its vertical tension root in the upper half and its diagonal shear tail in the lower half, it is canonically inverted so that the vertical root is at the bottom ($y \in [128, 255]$) and the diagonal bend propagates upward ($y \in [0, 128]$).

#### Assumption A2: Prohibition of Vertical Flip Augmentation During Training
* **Statement:** In `src/train_exp5_mechanisms_256.py`, `RandomVerticalFlip` is **completely removed** from the data augmentation pipeline; only `RandomHorizontalFlip(p=0.5)` is permitted.
* **Reason:** In Experiment 4, `RandomVerticalFlip(p=0.5)` was used. However, flipping a beam upside down turns a bottom-tension flexural crack into a top-edge crack, and turns an upward-curving flexure-shear crack into a downward-curving crack—destroying the physical distinction between tension-zone and compression-zone mechanics! By contrast, a horizontal flip (`left <-> right`) preserves bottom-edge tension initiation and simply mirrors a left-span shear crack ($+45^\circ$) into a right-span shear crack ($135^\circ$), which is physically valid by beam symmetry.

#### Assumption A3: Piecewise Skeleton Curvature ($\Delta \theta$) as the Discriminator for Flexure-Shear
* **Statement:** We split the bounding box of the primary crack component at its vertical midpoint $y_{\text{mid}} = \frac{y_{\min} + y_{\max}}{2}$ into a lower tension zone ($M_{\text{lower}}$) and an upper web zone ($M_{\text{upper}}$). A crack is classified as **`flexure_shear` (Class 2)** if and only if:
  1. Total span $L_{\max} \ge 65\text{ px}$ and continuity ratio $C_{\text{ratio}} \ge 0.52$.
  2. One half exhibits vertical orientation ($\theta \in [66^\circ, 114^\circ]$) while the other half exhibits diagonal orientation ($\theta \in [21^\circ, 69^\circ]$ or $[111^\circ, 159^\circ]$).
  3. The piecewise angular deflection between the two halves satisfies $\Delta \theta = |\theta_{\text{lower}} - \theta_{\text{upper}}| \ge 16.0^\circ$.
* **Reason:** Under combined bending and shear, a crack initiates vertically where flexural tensile stress $\sigma_x = \frac{My}{I}$ is maximum (at the bottom fiber, where shear stress $\tau_{xy} = 0$). As the crack tip propagates upward toward the neutral axis, $\sigma_x$ decreases while shear stress $\tau_{xy} = \frac{VQ}{Ib}$ increases, rotating the principal tensile stress trajectory toward $45^\circ$. A piecewise bend of $\ge 16^\circ$ mathematically captures this trajectory rotation.

#### Assumption A4: Euclidean Distance Transform (EDT) Width Taper for Flexural Cracks
* **Statement:** For every pixel $(x, y)$ inside the main crack mask, we compute the Euclidean Distance Transform $\text{EDT}(x, y)$, which measures the distance to the nearest crack boundary. The local aperture half-width in the lower vs. upper half is estimated via the 80th percentile of $\text{EDT}$:
  $$r_{\text{taper}} = \frac{P_{80}(\text{EDT}_{\text{lower}})}{P_{80}(\text{EDT}_{\text{upper}})}$$
  A vertical crack ($\theta \in [69^\circ, 111^\circ]$, $\Delta \theta < 20^\circ$) is classified as **`flexural` (Class 1)** and oriented such that $r_{\text{taper}} \ge 1.0$ (widest at the bottom tension fiber, tapering upward).
* **Reason:** By the Bernoulli-Euler hypothesis (plane sections remain plane), tensile strain $\varepsilon_x(y) = \kappa (y - y_{\text{NA}})$ varies linearly with distance from the neutral axis. Therefore, a flexural crack must have its widest crack mouth opening displacement (CMOD) at the bottom tension face and taper toward zero at the neutral axis.

#### Assumption A5: Single-Trajectory vs. Parallel Echelon Multiplicity for Web-Shear vs. Torsion
* **Statement:**
  * **`web_shear` (Class 3):** Defined by a single dominant diagonal fracture ($\theta \in [21^\circ, 69^\circ]$ or $[111^\circ, 159^\circ]$, $L_{\max} \ge 60\text{ px}$, $C_{\text{ratio}} \ge 0.52$) without parallel diagonal companions.
  * **`torsion` (Class 4):** Defined by $\ge 2$ significant elongated diagonal components ($L_k \ge 38\text{ px}$, elongation $\sqrt{\lambda_1 / \lambda_2} \ge 2.2$, $\theta_k$ diagonal) that run in parallel echelon formation ($|\theta_1 - \theta_2| \le 24^\circ$) with intermediate continuity ratio $0.25 \le C_{\text{ratio}} \le 0.85$.
* **Reason:** Pure web-shear failure typically localizes into a single dominant critical diagonal tension crack across the web. Pure torsion generates a uniform circulatory shear stress field around the perimeter, producing a family of parallel helical diagonal cracks spaced regularly across the face.

#### Assumption A6: Longitudinal Rebar Alignment for Corrosion Splitting
* **Statement:** A continuous horizontal fracture ($\theta \le 21^\circ$ or $\theta \ge 159^\circ$, $\Delta \theta < 20^\circ$, $L_{\max} \ge 60\text{ px}$, $C_{\text{ratio}} \ge 0.52$) is classified as **`corrosion_splitting` (Class 5)**.
* **Reason:** When longitudinal reinforcing steel corrodes, the iron oxide corrosion products expand to $2\times\text{--}6\times$ the original steel volume, exerting radial bursting pressure against the concrete cover and splitting it in a straight line parallel to the horizontal rebar axis.

#### Assumption A7: Multi-Component Isotropic Dispersion for Shrinkage Crazing
* **Statement:** Patches containing $\ge 4$ significant crack components ($\ge 30\text{ px}$ area, $\ge 25\text{ px}$ span) where no single component dominates ($C_{\text{ratio}} < 0.48$) are classified as **`shrinkage_map` (Class 6)**.
* **Reason:** Plastic and drying shrinkage create an isotropic biaxial surface tensile stress field ($\sigma_x \approx \sigma_y$), resulting in an interconnected polygonal mesh (map cracking) rather than a single directional structural crack.

---

## 3. Step-by-Step Engineering Log

### Step 1: Audit of Raw Datasets & Bounding-Box Guided Mask Extraction
* **Action:** Created [`src/prepare_experiment5_mechanisms.py`](file:///D:/Downloads/concrete_ddpm/src/prepare_experiment5_mechanisms.py) combining four complementary raw data sources:
  1. `DeepCrack` (`train_img` + `test_img`, 537 images at $544 \times 384$ with pixel masks, $256 \times 256$ sliding window at stride 64).
  2. `CrackForest` (118 images at $480 \times 320$ with `.mat` pixel segmentation and boundary masks, stride 64).
  3. `RC_10519` (`Reinforced concrete structure crack detection data`, 6,000 reinforced concrete structural images with LabelMe JSON bounding boxes).
  4. `SDNET2018` (3,500 uncracked concrete deck, pavement, and wall patches for Class 0).
* **Reason for Box-Guided Masking on RC_10519:** In Experiment 3, RC_10519 bounding boxes were found to lack pixel masks. However, throwing away 10,519 real reinforced concrete structure photos loses valuable structural crack shapes. By intersecting adaptive local dark-fissure thresholding (`local_mean - gray > 15.0`) **strictly with the interior of the human-labeled bounding boxes**, background clutter outside the annotated crack is zeroed out, yielding clean pixel masks of real reinforced concrete cracks.

### Step 2: Transfer Learning Architecture & Semantic Embedding Initialization
* **Action:** Created [`src/train_exp5_mechanisms_256.py`](file:///D:/Downloads/concrete_ddpm/src/train_exp5_mechanisms_256.py) using `AttentionUNet256` with `num_classes=7` (`base_channels=40`, 13,820,323 parameters).
* **Reason for Transfer Learning from Experiment 4:**
  * Experiment 4 already trained `AttentionUNet256` for 50 epochs (109,500 iterations, ~8.5 hours) to master $256 \times 256$ concrete aggregate textures, multi-scale denoising, and FlashAttention crack continuity.
  * Every single layer in the 7-class network has the **exact same tensor shape** as the 5-class network, except `class_embed.weight` which expands from `(6, 160)` to `(8, 160)`.
  * We initialize `class_embed.weight` using a **Semantic Mechanics Projection**:
    - $\mathbf{e}_{\text{non\_cracked}}^{(5)} \leftarrow \mathbf{e}_{\text{non\_cracked}}^{(4)}$
    - $\mathbf{e}_{\text{flexural}}^{(5)} \leftarrow \mathbf{e}_{\text{transverse}}^{(4)}$
    - $\mathbf{e}_{\text{flexure\_shear}}^{(5)} \leftarrow 0.5\,\mathbf{e}_{\text{transverse}}^{(4)} + 0.5\,\mathbf{e}_{\text{diagonal}}^{(4)}$
    - $\mathbf{e}_{\text{web\_shear}}^{(5)} \leftarrow \mathbf{e}_{\text{diagonal}}^{(4)}$
    - $\mathbf{e}_{\text{torsion}}^{(5)} \leftarrow 0.7\,\mathbf{e}_{\text{diagonal}}^{(4)} + 0.3\,\mathbf{e}_{\text{alligator}}^{(4)}$
    - $\mathbf{e}_{\text{corrosion\_splitting}}^{(5)} \leftarrow \mathbf{e}_{\text{longitudinal}}^{(4)}$
    - $\mathbf{e}_{\text{shrinkage\_map}}^{(5)} \leftarrow \mathbf{e}_{\text{alligator}}^{(4)}$
    - $\mathbf{e}_{\emptyset}^{(5)} \leftarrow \mathbf{e}_{\emptyset}^{(4)}$ (CFG unconditional token)
  * This gives the model a warm start with zero cold-start shock, allowing 25 fine-tuning epochs ($\text{lr}=1.0 \times 10^{-4}$) to specialize the embeddings and attention maps onto the 7 structural mechanisms.

### Step 3: Class-Balanced Sampling & Hardware Resource Controls
* **Action:** Configured `WeightedRandomSampler` across all 7 classes, batch size 6, FP16 AMP (`torch.amp.autocast`), gradient clipping at 1.0, and $15\%$ Classifier-Free Guidance label dropout.
* **Reason:** Ensures every mini-batch presents an equal 14.3% probability ($1/7$) for each structural failure mechanism regardless of raw image counts, while staying strictly within ~3.5 GB VRAM and $\le 3.2\text{ GB}$ system RAM on the RTX 4050 Laptop GPU.

### Step 4: Phase 1 Dataset Curation Results & Visual Verification Audit
* **Action:** Executed [`src/prepare_experiment5_mechanisms.py`](file:///D:/Downloads/concrete_ddpm/src/prepare_experiment5_mechanisms.py) across `DeepCrack`, `CrackForest`, `RC_10519`, and `SDNET2018`, writing **15,440 native $256 \times 256$ images** to `D:\Downloads\concrete_ddpm\data\processed_exp5_mechanisms\train\` and manifest `D:\Downloads\concrete_ddpm\data\processed_exp5_mechanisms\train.csv`.
* **Curated Class Distribution & Imbalance Reduction:**
  - `Class 0 (non_cracked)`: **3,500** patches (`22.67%`) — Uncracked concrete surfaces (decks, walls, pavements)
  - `Class 1 (flexural)`: **944** patches (`6.11%`) — Bottom-tension initiated vertical cracks ($\theta \in [69^\circ, 111^\circ]$), widest at bottom ($r_{\text{taper}} \ge 1.0$)
  - `Class 2 (flexure_shear)`: **816** patches (`5.28%`) — Combined $M + V$ cracks starting vertically at the bottom tension zone and curving diagonally upward ($\Delta \theta \ge 16^\circ$)
  - `Class 3 (web_shear)`: **2,742** patches (`17.76%`) — Single dominant diagonal tension fractures ($\theta \in [21^\circ, 69^\circ] \cup [111^\circ, 159^\circ]$)
  - `Class 4 (torsion)`: **1,634** patches (`10.58%`) — Multi-strand parallel echelon diagonal fractures ($\ge 2$ parallel diagonal cracks with $|\theta_1 - \theta_2| \le 24^\circ$)
  - `Class 5 (corrosion_splitting)`: **3,506** patches (`22.71%`) — Continuous horizontal splitting cracks along longitudinal reinforcement ($\theta \le 21^\circ$ or $\ge 159^\circ$)
  - `Class 6 (shrinkage_map)`: **2,298** patches (`14.88%`) — Interconnected multi-directional polygonal crazing networks ($\ge 4$ components, $C_{\text{ratio}} < 0.48$)
* **Reason this improves over Experiment 4:** In Experiment 4, the rarest crack class had only 302 patches (`18.2:1` imbalance ratio). By incorporating `DeepCrack` `test_img` at stride 64 and bounding-box guided mask extraction on `RC_10519`, the dataset expanded from 13,136 to **15,440 patches**, and the maximum-to-minimum class imbalance dropped more than $4\times$ to **`4.3:1`** (`3,506` vs `816`), with even the rarest structural mechanism (`flexure_shear`) having **816 distinct patches**.
* **Visual Verification Audit (`dataset_verification_grid.png`):** Generated a $7 \times 8$ visual contact sheet (`D:\Downloads\concrete_ddpm\experiments\structural_mechanism_ddpm\dataset_verification_grid.png`) and inspected all 7 rows to verify physical correspondence before training.

### Step 5: Phase 3 Execution — 25-Epoch Transfer-Learning Training (`task-968`)
* **Action:** Executed `src/train_exp5_mechanisms_256.py --epochs 25 --batch-size 6` (`64,350` parameter update iterations across `15,440` curated patches).
* **Convergence & Performance:**
  * Warm-starting from `crack_type_256_epoch_050.pt` via semantic mechanics embedding projection resulted in an Epoch 1 mean MSE loss of **`0.013176`** (compared to `0.054442` in Experiment 4 Epoch 1 from scratch—a $4.13\times$ lower starting error).
  * Across Epochs 1–25, mean MSE loss converged smoothly to **`0.012985`** at Epoch 25 (reaching a minimum of **`0.012607`** at Epoch 19), outperforming Experiment 4's final 50-epoch loss (`0.014690`).
  * Average epoch duration was `~625s–822s` (`~3.3–4.0 it/s`), completing all 25 checkpoints (`exp5_mechanism_256_epoch_001.pt` through `exp5_mechanism_256_epoch_025.pt`).

### Step 6: Phase 4 Execution — 50-Step DDIM + CFG ($w=2.5$) Sampling & Mechanics Evaluation (`task-993`, `task-997`)
* **Action:** Generated 16 native $256 \times 256$ images per class (**112 images total**) across all 7 structural failure mechanisms using `src/sample_exp5_mechanisms_256.py` and evaluated them with `src/evaluate_exp5_mechanisms_256.py`.
* **Key Quantitative Findings:**
  * **Inception Fidelity:** Achieved **`KID = 0.03931 ± 0.01168`** (a **13.1% improvement** over Experiment 4's `0.04526 ± 0.01413`, and the best KID score across all 5 experiments in the project) alongside **`FID = 170.582`**.
  * **Mechanism-Specific Structural Differentiation:**
    - `Class 0 (non_cracked)`: **93.8% clean surface rate** (15/16 completely free of dark crack fissures).
    - `Class 1 (flexural)`: **93.8% structural mechanism adherence**, longest mean crack span ($L_{\max} = 111.3\text{ px}$), vertical orientation, and bottom-rooted tension initiation.
    - `Class 2 (flexure_shear)`: Highest piecewise skeleton curvature across all classes (**$\Delta \theta = 21.4^\circ$** vs. $8.4^\circ$ for corrosion splitting) and positive bottom-to-top aperture taper (**$r_{\text{taper}} = 1.05$**), confirming combined bending-and-shear trajectory rotation.
    - `Class 3 (web_shear)`: **93.8% crack presence** and **62.5% strict single-diagonal adherence** ($L_{\max} = 88.1\text{ px}$, $\Delta \theta = 14.3^\circ$).
    - `Class 4 (torsion)`: **93.8% echelon multiplicity adherence** ($L_{\max} = 105.9\text{ px}$), synthesizing multi-strand parallel diagonal shear-flow fractures.
    - `Class 5 (corrosion_splitting)`: Lowest curvature (**$\Delta \theta = 8.4^\circ$**) and highest single-component continuity ratio (**$C_{\text{ratio}} = 0.409$**), reflecting straight horizontal splitting along longitudinal reinforcement.
    - `Class 6 (shrinkage_map)`: **100.0% mechanism adherence** with low single-component concentration ($C_{\text{ratio}} = 0.268$, $14.92$ average components), confirming isotropic polygonal map crazing.
  * **Recorded Observation — Chromatic Color Shifts Across Generated Tiles:**
    - Visual inspection of the Experiment 5 generated grids showed that individual tiles exhibited distinct color tints (blue, green, purple/violet, and warm brown).
    - **Root Causes:** (1) Independent 3-channel Gaussian noise initialization $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ allows per-channel DC means to drift apart at high $t$; (2) Classifier-Free Guidance ($w = 2.5$) multiplies any slight RGB channel offset between conditional and unconditional predictions by $250\%$ across 50 sequential DDIM steps; and (3) uncalibrated field illumination in `RC_10519` and `DeepCrack` exposed the model to colored ambient lighting.
    - **Resolution:** Formally recorded in both this ledger and `EXPERIMENT_5_FULL_REPORT.md` (Sections 10 & 13), and resolved in Experiment 6 via **Assumption C4 (Chromatic White-Balance Normalization)**.


