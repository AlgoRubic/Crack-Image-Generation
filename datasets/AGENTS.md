# AGENTS.md — Datasets Domain (`datasets/`)

1. **Read-Only Manifests:** The `train.csv` files inside each subfolder of `datasets/` are canonical copies of the processed dataset manifests in `data/processed*/train.csv`.
2. **Relative Image Paths:** Every `path` column entry in `train.csv` is relative to the repository root (`D:\Downloads\concrete_ddpm\data\...`). Do not alter relative path prefixes inside `train.csv`.
