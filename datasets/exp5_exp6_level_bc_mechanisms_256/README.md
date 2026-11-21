# Dataset for Experiments 5 (Level B) & 6 (Level C): 7-Class Structural Failure Mechanism Dataset ($256 \times 256$)

- **Manifest File:** `train.csv` (`15,440` native $256 \times 256$ records)
- **Image Binary Location:** `D:\Downloads\concrete_ddpm\data\processed_exp5_mechanisms\train\`
- **Source Datasets (4 Combined Sources):**
  1. `DeepCrack`: `6,344` pixel-mask verified crops
  2. `RC_10519`: `4,832` human JSON bounding-box guided images
  3. `SDNET2018`: `3,500` clean non-cracked concrete tiles
  4. `CrackForest (CFD)`: `764` ground-truth `.mat` segmentation crops
- **7 Reinforced-Concrete Structural Failure Mechanism Classes:**
  - `0: non_cracked` (`3,500` images)
  - `1: flexural` (`944` images — canonically oriented with tension root at bottom edge $y=255$)
  - `2: flexure_shear` (`816` images — vertical lower root curving diagonally into upper web, $\Delta\theta \ge 16^\circ$)
  - `3: web_shear` (`2,742` images — inclined diagonal tension crack in web)
  - `4: torsion` (`1,634` images — $\ge 2$ parallel echelon diagonal fractures)
  - `5: corrosion_splitting` (`3,506` images — straight horizontal splitting along rebar axis)
  - `6: shrinkage_map` (`2,298` images — multi-directional polygonal crazing network)
