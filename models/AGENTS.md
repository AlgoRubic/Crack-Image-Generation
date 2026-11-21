# AGENTS.md — Models & Weights Domain (`models/`)

1. **Immutable Pre-Trained Checkpoints:** All `.pt` files inside `models/weights/` are finalized, converged model weights representing 90+ cumulative epochs of GPU training. Never delete or overwrite them.
2. **Loading Convention:** Each `.pt` file is a dictionary containing `ckpt["model"]` (the PyTorch `state_dict`).
