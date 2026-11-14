# Concrete Crack Image Generation

Research code and technical records for progressive diffusion-model experiments on synthetic concrete-crack image generation.

The project progresses from an unconditional 64×64 DDPM to binary crack-status conditioning, 128×128 augmentation, native-resolution morphology conditioning, seven-class structural-mechanism conditioning, and finally explicit spatial-physics trajectory/aperture conditioning at 256×256.

## Experiments

1. **Unconditional DDPM** — baseline concrete texture synthesis.
2. **Crack-status conditional DDPM** — cracked versus non-cracked control.
3. **128×128 augmented conditional DDPM** — higher resolution with RC_10519 cracked-image augmentation; evaluation intentionally closed before final quality metrics.
4. **Morphological crack-type conditioning** — five native-resolution morphology states.
5. **Structural failure mechanism conditioning** — seven physics-grounded mechanism classes.
6. **Physics-guided spatial control** — trajectory and aperture-field conditioning with `PhysicsControlUNet256`.

Each experiment folder contains the available status records, reports, metrics, logs, generated result grids, and implementation artifacts.

## Research website

The complete interactive archive is included under [`website/`](website/) and is publicly hosted at:

<https://concrete-crack-synthesis.beigetoad12.chatgpt.site>

To run the static site locally:

```powershell
cd website
python -m http.server 4173 --bind 127.0.0.1
```

Then open <http://127.0.0.1:4173/>.

## Code setup

Create a fresh Python environment and install the dependencies appropriate for your CUDA/PyTorch setup. The checked-in source scripts are in [`src/`](src/), including training, sampling, evaluation, dataset preparation, and the Experiment 6 physics trajectory solver.

Example entry points:

```powershell
python src/train.py
python src/train_conditional.py
python src/train_exp5_mechanisms_256.py
python src/train_exp6_spatial_physics_256.py
```

Review each experiment's report and status file before running a long training job.

## Large files intentionally excluded

The repository excludes the local virtual environment, raw datasets, model checkpoints, per-sample image dumps, and physics-mask arrays. These are many gigabytes and are not suitable for ordinary GitHub storage; raw datasets should be obtained from their original sources, and checkpoints can be regenerated from the training scripts. Curated result grids, reports, metrics, logs, plots, and presentations are retained.

## Evidence and limitations

FID, KID, LPIPS, crack-presence, continuity, trajectory-recall, and mechanism-adherence values are protocol-specific diagnostics. They do not establish structural safety or full-member physical validity. The technical reports document assumptions, sources of error, and the distinction between measured evidence and interpretation.
