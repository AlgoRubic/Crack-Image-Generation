# Dataset for Experiments 1 & 2: `SDNET2018` ($64 \times 64$)

- **Manifest File:** `train.csv` (`50,482` training records) and `validation.csv`
- **Image Binary Location:** `D:\Downloads\concrete_ddpm\data\processed\train\`
- **Source Dataset:** `SDNET2018` (Concrete Bridge Decks `D`, Pavements `P`, and Walls `W`).
- **Class Distribution:**
  - `non_cracked`: `42,846` images (`84.87%`)
  - `cracked`: `7,636` images (`15.13%`)
- **How It Was Used:**
  - **Experiment 1:** Trained unconditionally on all `50,482` images.
  - **Experiment 2:** Trained conditionally (`0: non_cracked`, `1: cracked`) using `WeightedRandomSampler` to balance the $5.6:1$ class ratio.
