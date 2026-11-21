# Dataset for Experiment 4 (Level A): 5-Class Morphological Crack Dataset ($256 \times 256$)

- **Manifest File:** `train.csv` (`13,139` native $256 \times 256$ records)
- **Image Binary Location:** `D:\Downloads\concrete_ddpm\data\processed_256\train\`
- **Source Datasets:**
  1. `DeepCrack` (`2,066` high-resolution $256 \times 256$ sliding-window crops with ground-truth binary masks)
  2. `SDNET2018` (`6,000` subsampled non-cracked concrete tiles + `5,073` continuity-verified cracked tiles)
- **5 Morphological Classes:**
  - `0: non_cracked` (`6,000` images)
  - `1: longitudinal` (`978` images, $\theta \approx 0^\circ \pm 22.5^\circ$)
  - `2: transverse` (`335` images, $\theta \approx 90^\circ \pm 22.5^\circ$)
  - `3: diagonal` (`686` images, $\theta \approx 45^\circ / 135^\circ \pm 22.5^\circ$)
  - `4: alligator_map` (`5,140` images, $\ge 4$ branches, $C_{\text{ratio}} < 0.50$)
