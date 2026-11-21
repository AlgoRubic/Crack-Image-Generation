# AGENTS.md — `pipeline/core/`

- Any changes to `diffusion.py`, `model.py`, or `model_attention.py` affect checkpoint loading compatibility across all 6 experiments. Do not rename state_dict layer keys (`init_conv`, `class_embed`, `time_mlp`, `downs`, `ups`, `final_conv`).
