# Dataset for Experiment 3: Augmented `SDNET2018 + RC_10519` ($128 \times 128$)

- **Manifest File:** `train.csv` (`61,001` records)
- **Image Binary Location:** `D:\Downloads\concrete_ddpm\data\processed_128\train\`
- **Source Datasets:**
  1. `SDNET2018`: `50,482` images (`42,846` non-cracked + `7,636` cracked)
  2. `RC_10519` (*Reinforced Concrete Structure Crack Detection Data*): `10,519` cracked images
- **Class Distribution:**
  - `0: non_cracked`: `42,846` images
  - `1: cracked`: `18,155` images ($2.38\times$ more cracked samples than Exp 1 & 2)

The matching publisher record is [Wang (2026), Mendeley Data v1](https://data.mendeley.com/datasets/3ghxck2wd5/1), DOI `10.17632/3ghxck2wd5.1`. The publisher requests the [Wang et al. (2024) paper](https://onlinelibrary.wiley.com/doi/10.1002/suco.202300937) as well. Both citations are included in the project bibliography. The local archive has not been matched to a publisher-provided checksum, so identifying this record does not prove the original download history.

The publisher describes crack and intact classes. This project's local subset and preparation script assign all 10,519 RC images to `cracked`; that assignment needs to be distinguished from independently verified labels. Run `python pipeline/audit_rc10519.py` for local archive/extraction and annotation-geometry checks. Results and attribution metadata are in `datasets/source_provenance/`. Bounding boxes establish neither pixel-level masks nor structural failure mechanisms. Perceptual screening also found apparent overlap with other named datasets; use the v5 connected groups rather than dataset names to isolate prospective splits.
