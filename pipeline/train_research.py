"""Clean 256px research training lineage, separate from historical checkpoints.

python pipeline/train_research.py --experiment 4 --run-id exp4-seed2026 --epochs 50
Use --smoke for one training/validation batch; its weights are never production models.
"""
import argparse
import copy
import csv
import hashlib
import json
import importlib.metadata
import shutil
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model_attention import AttentionUNet256
from diffusion import GaussianDiffusion
from train_exp6_spatial_physics_256 import normalize_chromatic_cast, extract_paired_spatial_physics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_manifests(folder):
    rows = {s: read_rows(folder / (s + '.csv')) for s in ('train', 'validation', 'test')}
    seen = {}
    for split, items in rows.items():
        if not items:
            raise ValueError(f'Empty {split} manifest')
        for row in items:
            identities = ['content:' + row['original_sha256']]
            if row.get('acquisition_group'):
                identities.append('acquisition:' + row['acquisition_group'])
            for group in identities:
                if seen.setdefault(group, split) != split:
                    raise ValueError('Image-content or acquisition leakage between splits')
            if row['split'] != split:
                raise ValueError('Manifest row split disagrees with filename')
    return rows


class Images(Dataset):
    def __init__(self, rows, spatial, augment):
        self.rows, self.spatial, self.augment = rows, spatial, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as image:
            rgb = np.array(image.convert('RGB'))
        if rgb.shape != (256, 256, 3):
            raise ValueError(f'Unexpected native resolution: {row["path"]}')
        if self.augment and random.random() < .5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        # Identical image preprocessing for Exp4/5/6 isolates spatial conditioning.
        rgb = normalize_chromatic_cast(rgb)
        cid = int(row['class_id'])
        phys = extract_paired_spatial_physics(rgb, cid) if self.spatial else np.zeros((2, 256, 256), np.float32)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 127.5 - 1, torch.from_numpy(phys) * 2 - 1, cid


def run(args):
    args.conditioning=args.conditioning or ('spatial' if args.experiment==6 else 'class_only')
    if args.experiment!=6 and args.conditioning!='class_only':
        raise ValueError('Spatial conditioning is available only for Experiment 6')
    if not 1 <= args.batch_size <= 8 or args.epochs < 1:
        raise ValueError('Batch size must be 1..8 and epochs positive')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Run ID must be a single directory name')
    if not 0 < args.ema_decay < 1 or args.lr <= 0:
        raise ValueError('EMA decay must be between 0 and 1; learning rate must be positive')
    if not args.smoke and args.split_version!='research_v5':
        raise ValueError('Full clean training requires research_v5 similarity-isolated splits; older versions are archival/smoke only')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = 'exp4_level_a_morphology_256' if args.experiment == 4 else 'exp5_exp6_level_bc_mechanisms_256'
    folder = ROOT / 'datasets' / dataset / args.split_version
    rows = verify_manifests(folder)
    run_dir = ROOT / 'experiments' / ('crack_type_conditional' if args.experiment == 4 else
              'structural_mechanism_ddpm' if args.experiment == 5 else 'exp6_level_c_spatial_physics') / 'research_runs' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot=run_dir/'source_snapshot';snapshot.mkdir()
    source_files=[Path(__file__),ROOT/'src/model_attention.py',ROOT/'src/diffusion.py',ROOT/'src/train_exp6_spatial_physics_256.py']
    for source in source_files:
        shutil.copyfile(source,snapshot/source.name)
    config = dict(vars(args), initialization='scratch', transfer_ancestors=[],
                  manifests={s: {'path': str(folder / (s + '.csv')), 'sha256': sha(folder / (s + '.csv'))} for s in rows},
                  script_sha256=sha(Path(__file__)), validation_protocol='Fixed noise/timestep seed, no augmentation, full validation each epoch',
                  test_policy='Manifest isolation checked; test images never loaded by trainer',
                  label_limitation='Geometry-derived proxy labels; no independent expert validation',
                  smoke_only=args.smoke, validation_weights='ema', best_checkpoint_inference_key='ema',
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  python=sys.version, packages={p.metadata['Name']:p.version for p in importlib.metadata.distributions()})
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spatial = args.experiment == 6
    classes = 5 if args.experiment == 4 else 7
    model = AttentionUNet256(in_channels=5 if spatial else 3, base_channels=40,
                             num_classes=classes, cfg_dropout=0.).to(device)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    diffusion = GaussianDiffusion(device=device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(Images(rows[s], spatial and args.conditioning=='spatial', s == 'train'), batch_size=args.batch_size,
                            shuffle=s == 'train', generator=generator if s == 'train' else None,
                            num_workers=0) for s in ('train', 'validation')}
    best = float('inf')
    start_epoch = 1
    if args.resume:
        # Resume only checkpoints created by this trusted local research trainer.
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous = state['config']
        if previous.get('initialization') != 'scratch' or previous.get('transfer_ancestors'):
            raise ValueError('Resume checkpoint does not have a clean scratch lineage')
        if previous['smoke_only'] != args.smoke:
            raise ValueError('Smoke checkpoints cannot seed full research runs')
        for key in ('experiment', 'conditioning', 'seed', 'batch_size', 'ema_decay', 'lr', 'manifests', 'script_sha256'):
            if previous.get(key) != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optim.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        generator.set_state(state['rng']['loader'])
        start_epoch = state['epoch'] + 1
        best = state['best_validation_loss']
        config['resume_checkpoint_sha256'] = sha(Path(args.resume))
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    final_epoch = args.smoke_epochs if args.smoke else args.epochs
    if start_epoch > final_epoch:
        raise ValueError('No remaining epochs requested')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with (run_dir / 'loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'validation_loss', 'seconds', 'peak_gpu_gib'])
        writer.writeheader()
        for epoch in range(start_epoch, final_epoch + 1):
            start = time.perf_counter()
            losses = {}
            for phase in ('train', 'validation'):
                training = phase == 'train'
                model.train(training)
                net = model if training else ema
                total, count = 0., 0
                val_rng = torch.Generator(device=device).manual_seed(100000 + args.seed)
                with torch.set_grad_enabled(training):
                    for step, (image, phys, labels) in enumerate(loaders[phase]):
                        image, phys, labels = image.to(device), phys.to(device), labels.to(device)
                        rng = None if training else val_rng
                        t = torch.randint(0, 1000, (len(image),), device=device, generator=rng)
                        noise = torch.randn(image.shape, device=device, generator=rng)
                        noisy, _ = diffusion.add_noise(image, t, noise)
                        if training:
                            drop = torch.rand(len(image), device=device) < .15
                            labels = torch.where(drop, classes, labels)
                            phys = torch.where(drop[:, None, None, None], -1., phys)
                        inputs = torch.cat((noisy, phys), 1) if spatial else noisy
                        with torch.amp.autocast('cuda', enabled=device.type == 'cuda', dtype=torch.float16):
                            pred = net(inputs, t, labels, force_dropout=False)
                            loss = torch.nn.functional.mse_loss(pred.float(), noise)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite loss')
                        if training:
                            optim.zero_grad(set_to_none=True)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optim)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            scale_before = scaler.get_scale()
                            scaler.step(optim)
                            scaler.update()
                            if scaler.get_scale() >= scale_before:
                                with torch.no_grad():
                                    for e, p in zip(ema.parameters(), model.parameters()):
                                        e.lerp_(p, 1 - args.ema_decay)
                        total += loss.item() * len(image)
                        count += len(image)
                        if args.smoke:
                            break
                losses[phase] = total / count
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0
            if peak > 4.:
                raise RuntimeError(f'GPU peak {peak:.2f} GiB exceeds configured 4 GiB budget')
            record = dict(epoch=epoch, train_loss=losses['train'], validation_loss=losses['validation'],
                          seconds=time.perf_counter()-start, peak_gpu_gib=peak)
            writer.writerow(record)
            stream.flush()
            print(json.dumps(record), flush=True)
            improved = losses['validation'] < best
            best = min(best, losses['validation'])
            state = dict(epoch=epoch, model=model.state_dict(), ema=ema.state_dict(), optimizer=optim.state_dict(),
                         scaler=scaler.state_dict(), config=config, validation_loss=losses['validation'],
                         best_validation_loss=best,
                         rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                  cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                                  loader=generator.get_state()))
            torch.save(state, run_dir / 'last.pt')
            if improved:
                torch.save(state, run_dir / 'best.pt')
    (run_dir / 'status.json').write_text(json.dumps(dict(status='completed', smoke_only=args.smoke, best_validation_loss=best)), encoding='utf-8')
    print(f'Run saved: {run_dir}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=int, choices=[4, 5, 6], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, choices=[1, 2], default=1,
                        help='One batch per phase per epoch; 2 permits resume regression checks')
    parser.add_argument('--resume', help='Trusted local research checkpoint; validates clean lineage and matching manifests')
    parser.add_argument('--split-version', choices=['research_v1','research_v2','research_v3','research_v4','research_v5'], default='research_v5',
                        help='v5 also conservatively groups all declared-threshold perceptual similarity candidates')
    parser.add_argument('--conditioning',choices=['class_only','spatial'],default=None,
                        help='Exp6 class_only keeps the same five-channel model but uses constant null spatial fields')
    run(parser.parse_args())
