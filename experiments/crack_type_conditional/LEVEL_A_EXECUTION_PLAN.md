# Level A Execution Plan: Morphological Crack-Type Conditioned Generation at Native 256×256

**Author:** Antigravity  
**Project:** Synthetic Concrete-Crack Image Generation Research (`D:\Downloads\concrete_ddpm`)  
**Date:** 2026-11-07 (Updated & Finalized)  
**Artifact Location:** `experiments/crack_type_conditional/LEVEL_A_EXECUTION_PLAN.md`  

---

## 1. Project Objective & Strategic Vision

The goal of **Level A** is to synthesize **realistic, continuous, and geometrically controllable concrete crack images** at native **$256 \times 256$ resolution**, conditioned on structural crack morphology:

```text
Class 0: non_cracked       (clean concrete surface, aggregate textures)
Class 1: longitudinal      (horizontal linear crack, orientation ~ 0° ± 20°)
Class 2: transverse        (vertical linear crack, orientation ~ 90° ± 20°)
Class 3: diagonal          (shear/torsion proxy, orientation ~ 45° ± 20° or 135° ± 20°)
Class 4: alligator_map     (interconnected polygonal network / crazing)
```

### Strategic Principles Informed by Previous Experiments:
1. **Native $256 \times 256$ Training:** Raw images will be cropped at native $256 \times 256$ resolution. We eliminate downsampling to $128 \times 128$ or $64 \times 64$, preserving $100\%$ of fine hairline crack connectivity.
2. **Global Attention in U-Net:** Introduce multi-head self-attention at intermediate spatial resolutions ($32 \times 32$ and $16 \times 16$) to eliminate crack fragmentation and allow edge-to-edge trajectory continuity.
3. **Classifier-Free Guidance (CFG):** Train with $10\%$ label dropout and sample with guidance scale $w = 2.5 \sim 3.0$ to overcome background concrete dominance.
4. **Fast DDIM Sampling:** 50–100 deterministic diffusion steps to generate high-fidelity samples in seconds rather than minutes.
5. **Realistic Multi-Tiered Continuity Filtering:** Rather than a rigid, artificial 120-pixel cutoff, we preserve genuine microcracks ($\ge 50$ px) and alligator network segments ($\ge 25$ px) while strictly filtering out non-crack speckle noise ($< 20$ px).

---

## 2. Dataset Strategy & Sourcing

### 2.1 Why Current Datasets Are Insufficient on Their Own
* **SDNET2018 (56,092 images):** Binary tags only (`cracked` vs `non_cracked`). No orientation, morphology, or width labels. Many crops are surface noise, bridge expansion joints, or surface voids.
* **RC_10519 (10,519 images):** All 16,157 annotations are **rectangular bounding boxes**, not pixel masks or skeleton lines. Many boxes contain surface spalling or blurred hairline sections rather than continuous structural crack lines.

### 2.2 Recommended High-Resolution Datasets

| Dataset | Images / Crops | Native Resolution | Annotation Type | Role in Level A |
|---|---|---|---|---|
| **Crack500** | 500 images | $2000 \times 1500$ | Pixel-level binary masks | **Primary:** Extract native $256 \times 256$ crops with verified continuous crack paths (transverse, longitudinal, diagonal). |
| **DeepCrack** | 537 images | $544 \times 384$ | Multi-scale pixel masks | **Primary:** High-contrast structural cracks on concrete walls and bridges with explicit orientation. |
| **DamSegment** | 3,000+ images | $1024 \times 1024+$ | Segmentation masks & damage classes | Structural concrete dam cracking with clean separation of spalling vs genuine fracture. |
| **dacl10k** | 10,000 images | High-res real-world | Multi-class damage masks | Outstanding source for `alligator_map` cracking and multi-directional bridge distress. |
| **SDNET2018** | Selected crops | $256 \times 256$ | Binary labels | Background aggregate texture and uncracked concrete baseline (Class 0). |

### 2.3 Dataset Composition & Multi-Tiered Continuity Rules

* **Target Dataset Size:** $\sim 25,000 - 30,000$ native $256 \times 256$ crops.
* **Target Distribution:**
  * Class 0 (`non_cracked`): 6,000 crops
  * Class 1 (`longitudinal`): 5,000 crops
  * Class 2 (`transverse`): 5,000 crops
  * Class 3 (`diagonal`): 5,000 crops
  * Class 4 (`alligator_map`): 4,000 crops

#### Multi-Tiered Continuity & Quality Filtering Rules:
Rather than a rigid single cutoff, we establish an engineering-grounded threshold based on crack type:

1. **Linear Structural Cracks (Classes 1, 2, 3):**
   * **Major Structural Cracks:** Longest continuous path $\ge 90$ pixels.
   * **Micro-cracks & Initiation Tips:** Continuous path between **$50$ and $90$ pixels**, provided they form a single coherent connected component (low fragmentation).
2. **Alligator / Map Cracking (Class 4):**
   * Individual segment length $\ge 25$ pixels, with $\ge 2$ interconnected junction nodes forming a closed or semi-closed network.
3. **Noise Rejection Filter:**
   * Any isolated dark feature or speckle cluster whose maximum continuous length is **$< 20$ pixels** is automatically rejected as non-crack noise (aggregate shadows, dust, or camera sensor noise).
4. **Continuity Ratio ($C_{\text{ratio}}$):**
   $$C_{\text{ratio}} = \frac{\text{Length of Longest Continuous Segment}}{\text{Total Dark Crack Pixels}} \ge 0.65$$
   Ensures that at least $65\%$ of the crack pixels belong to the primary crack trajectory, eliminating fragmented "salt-and-pepper" clusters.

---

## 3. Data Processing & Orientation Tagging Pipeline

```
Raw High-Resolution Datasets (Crack500, DeepCrack, dacl10k, SDNET2018)
                                │
                                ▼
            [ Native 256 x 256 Sliding-Window Cropping ]
                                │
                                ▼
                 [ Morphological Skeletonization ]
                   - Extract 1-pixel centerline skeleton
                   - Compute connected components
                   - Apply Tiered Continuity Filter:
                       * Linear cracks: path >= 50 px, C_ratio >= 0.65
                       * Alligator: junctions >= 2, segment >= 25 px
                       * Noise: discard if max path < 20 px
                                │
                                ▼
                 [ Orientation Angle Calculation ]
            Principal axis angle θ = 0.5 * arctan2(2μ11, μ20 - μ02)
                                │
      ┌─────────────────┬───────┴─────────┬─────────────────┐
      ▼                 ▼                 ▼                 ▼
 |θ| <= 20°        ||θ| - 90°| <= 20°  ||θ| - 45/135°| <= 20° Junctions >= 2
Longitudinal          Transverse           Diagonal         Alligator / Map
 (Class 1)             (Class 2)           (Class 3)           (Class 4)
                                │
                                ▼
              Clean Level A Metadata (`data/processed_256/train.csv`)
         (path, source, class_id, dominant_angle, max_length, c_ratio)
```

---

## 4. Model Architecture: `AttentionUNet256`

To solve the receptive-field limitation, we upgrade `SmallUNet` to `AttentionUNet256`:

```
Input Latent / Noise x_t (3, 256, 256)
   │
   ├── Conv2d (3 -> 64)
   ├── DownBlock 1: (64 -> 64, stride 2)    --> [64, 128, 128]
   ├── DownBlock 2: (64 -> 128, stride 2)   --> [128, 64, 64]
   ├── DownBlock 3 + Attention: (128 -> 256, stride 2) --> [256, 32, 32]  <-- Self-Attention Block 1
   ├── DownBlock 4 + Attention: (256 -> 512, stride 2) --> [512, 16, 16]  <-- Self-Attention Block 2
   │
   ├── MiddleBlock: (512 -> 512) with Self-Attention  --> [512, 16, 16]  <-- Global Receptive Field
   │
   ├── UpBlock 4 + Attention: (512+512 -> 256, upsample) --> [256, 32, 32]  <-- Spatial Coordination
   ├── UpBlock 3 + Attention: (256+256 -> 128, upsample) --> [128, 64, 64]
   ├── UpBlock 2: (128+128 -> 64, upsample)  --> [64, 128, 128]
   ├── UpBlock 1: (64+64 -> 64, upsample)    --> [64, 256, 256]
   │
   └── Output Conv2d (64 -> 3) --> Predicted Noise Output (3, 256, 256)
```

### Key Architectural Specifications:
* **Multi-Head Self-Attention:** 4 attention heads applied at $32 \times 32$ and $16 \times 16$ feature maps. This allows distant pixels across the entire image to communicate, ensuring crack trajectories stay connected from boundary to boundary.
* **5-Class Conditioning with Dropout:** Class embedding ($d=512$). During training, randomly replace the class label with an unconditional null token $\emptyset$ with probability $p = 0.10$.
* **Classifier-Free Guidance (CFG) at Inference:**
  $$\epsilon_{\text{guided}} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\big)$$
  Sampling with $w = 2.5 \sim 3.0$ forces the model to synthesize bold, connected fracture trajectories rather than retreating into flat concrete background.
* **Fast DDIM Sampling:** 50 deterministic reverse steps, generating a batch of 8 images in **~18–22 seconds** on the RTX 4050 GPU.

---

## 5. Training Specifications & Hardware Budget

* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Windows 11.
* **Project Directory:** Strictly inside `D:\Downloads\concrete_ddpm\experiments\crack_type_conditional\`.
* **Hyperparameters:**
  * Resolution: $256 \times 256$ RGB.
  * Batch Size: **8** (VRAM usage: $\sim 3.6$ GB, safely below the 6.0 GB VRAM limit, completely preventing PCIe host memory paging).
  * Precision: Mixed Precision (FP16 autocast with PyTorch `GradScaler`).
  * Optimizer: AdamW (learning rate $1.5 \times 10^{-4}$, weight decay $0.01$, cosine decay schedule).
  * Diffusion Steps: 1,000 steps during training (linear beta schedule $1 \times 10^{-4}$ to $0.02$).
  * Epochs: 50.
* **Estimated Runtime:**
  * $\sim 28,000$ training images $\div 8 = 3,500$ iterations/epoch.
  * At $\sim 0.08\,\text{s}$/iteration with FP16: $\approx 280\,\text{s}$ ($\sim 4.6\,\text{min}$) per epoch.
  * Total 50-epoch training: **$\approx 3.8 - 4.5$ GPU hours**.

---

## 6. Pre-Defined Evaluation Protocol

To rigorously evaluate the generated images and avoid unverified claims, the following metrics will be calculated and saved:

1. **Crack Continuity & Connectivity Metrics:**
   * **Longest Continuous Segment:** Must average $>120$ pixels for major cracks, and $>50$ pixels for microcracks.
   * **Connected Component Count:** Must average $\le 2.0$ components per image (confirming elimination of the fragmented salt-and-pepper defect).
   * **Continuity Ratio ($C_{\text{ratio}}$):** Must exceed $0.70$.
2. **Orientation Adherence Metric:**
   * An independent ResNet-18 classifier trained strictly on real training crack images.
   * Measure accuracy, precision, recall, and F1 on generated samples across all 5 classes.
3. **Class-Matched Distribution Metrics (FID & KID):**
   * Compute standard 2048-dim Inception FID and KID separately for each class against real validation images of the **same class** (cracked vs cracked, non-cracked vs non-cracked).
4. **Visual Contact Sheets:**
   * 16 samples per class compiled into high-resolution 4×4 contact grids ($1034 \times 1034$ PNGs) for visual validation of surface roughness, aggregate distribution, and crack edge sharpness.

---

## 7. Phased Implementation Roadmap

```
Phase 1: Dataset Acquisition & Filtering (Crack500, DeepCrack, dacl10k, SDNET)
         ├── Download & unpack into data/raw/
         ├── Sliding-window crop at native 256x256
         ├── Morphological skeletonization & tiered continuity filtering
         └── Automated orientation classification & CSV manifest generation
                                 │
Phase 2: Architecture Implementation (`src/model_attention.py`)
         ├── Implement AttentionUNet256 with 4-head self-attention
         └── Implement Classifier-Free Guidance (CFG) conditioning
                                 │
Phase 3: Model Training
         ├── 50 epochs with batch size 8 on RTX 4050 GPU (FP16 AMP)
         └── Save checkpoints and loss logs in experiments/crack_type_conditional/
                                 │
Phase 4: Sampling & Contact Grids
         ├── DDIM 50-step generation with CFG w=2.5
         └── Save individual PNGs and 4x4 grids for all 5 classes
                                 │
Phase 5: Formal Evaluation
         ├── Continuity metric analysis (component count & length)
         ├── Orientation adherence F1 score
         └── Class-matched FID / KID computation
                                 │
Phase 6: Final Reporting & Handoff
         └── Update STATUS.md and compile LEVEL_A_FINAL_REPORT.md
```

---

## 8. Summary of Guarantees

* **Storage Compliance:** All code, data, logs, and outputs remain strictly inside `D:\Downloads\concrete_ddpm\`. Nothing is placed on the `C:` drive or outside the project folder.
* **Continuity Guarantee:** Combines native $256 \times 256$ resolution, self-attention layers, and CFG amplification to eliminate the small-fragment defect.
* **Honest Metrics:** All reported FID, KID, and orientation F1 scores will be calculated from actual generated sets and saved directly to disk.
