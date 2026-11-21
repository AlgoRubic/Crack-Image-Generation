# AGENTS.md — Pipeline Domain (`pipeline/`)

1. **Inference First:** `pipeline/generate_cracks.py` automatically loads pre-trained weights from `models/weights/exp<N>_.../`. Never trigger training when a user asks to test or generate images.
2. **Mini-Batch Memory Safety:** Always keep sampling mini-batches at `chunk_size = 8` (or `sub_batch_size = 4`) for $256 \times 256$ models so generation of `200` or `1234` images runs in constant $<3.8\text{ GB}$ GPU VRAM on the RTX 4050.
