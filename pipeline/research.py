"""Research dataset provenance and prospective split preparation.

Run: python pipeline/research.py prepare
No training, checkpoint mutation, or canonical manifest edits are performed.
"""
import argparse
import csv
import hashlib
import json
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['path'])
        writer.writeheader()
        writer.writerows(rows)


def recover_index():
    """Invert exact historical filename recipes without regenerating images."""
    indexes = {4: defaultdict(set), 5: defaultdict(set)}
    def add(exp, tag, original):
        name = hashlib.md5(tag.encode('utf-8')).hexdigest()[:16] + '.png'
        indexes[exp][name].add(original.relative_to(ROOT).as_posix())
    raw = ROOT / 'data/raw'
    for split in ('train_img', 'test_img'):
        for path in sorted((raw / 'DeepCrack-master/dataset' / split).glob('*.*')):
            with Image.open(path) as im:
                w, h = im.size
            for y in range(0, h - 256 + 1, 64):
                for x in range(0, w - 256 + 1, 64):
                    add(5, f'deepcrack5_{split}_{path.stem}_{x}_{y}', path)
            if split == 'train_img':
                for y in range(0, h - 256 + 1, 96):
                    for x in range(0, w - 256 + 1, 96):
                        add(4, f'deepcrack_{path.stem}_{x}_{y}', path)
    for path in sorted((raw / 'CrackForest-dataset-master/image').glob('*.jpg')):
        with Image.open(path) as im:
            w, h = im.size
        for y in range(0, h - 256 + 1, 64):
            for x in range(0, w - 256 + 1, 64):
                add(5, f'cfd5_{path.stem}_{x}_{y}', path)
    for path in sorted((raw / 'SDNET2018/images').rglob('*.jpg')):
        add(4, f'sdnet_uncracked_{path.name}', path)
        add(4, f'sdnet_cracked_{path.name}', path)
        add(5, f'sdnet_nc5_{path.name}', path)
    base = raw / 'external_cracks/Reinforced concrete structure crack detection data'
    for jf in sorted((base / 'label_json').glob('*.json')):
        path = base / 'images' / (jf.stem + '.jpg')
        if not path.exists():
            info = json.loads(jf.read_text(encoding='utf-8'))
            path = base / 'images' / Path(info.get('imagePath', '')).name
        if path.is_file():
            add(5, f'rc10519_{jf.stem}', path)
    return indexes


def assign_split(group, seed):
    bucket = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if bucket < .8 else 'validation' if bucket < .9 else 'test'


def prepare(root=ROOT):
    if root != ROOT:
        raise ValueError('Explicit repository root required')
    indexes = recover_index()
    outputs = []
    hash_cache = {}
    for exp, dataset in ((4, 'exp4_level_a_morphology_256'), (5, 'exp5_exp6_level_bc_mechanisms_256')):
        manifest = ROOT / 'datasets' / dataset / 'train.csv'
        with manifest.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        resolved, unresolved = [], []
        for row in rows:
            candidates = indexes[exp].get(Path(row['path']).name, set())
            if len(candidates) != 1:
                unresolved.append(dict(row, provenance_error=f'{len(candidates)} candidate originals'))
                continue
            original = next(iter(candidates))
            if original not in hash_cache:
                hash_cache[original] = digest(ROOT / original)
            group = hash_cache[original]
            resolved.append(dict(row, original_path=original, original_sha256=group,
                                 split=assign_split(group, 2026), historical_training_exposure='yes'))
        out = ROOT / 'datasets' / dataset / 'research_v1'
        write_csv(out / 'provenance.csv', resolved)
        write_csv(out / 'unresolved.csv', unresolved)
        groups = {}
        for split in ('train', 'validation', 'test'):
            selected = [r for r in resolved if r['split'] == split]
            write_csv(out / (split + '.csv'), selected)
            groups[split] = {r['original_sha256'] for r in selected}
        overlaps = sum(len(groups[a] & groups[b]) for a, b in
                       [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
        coverage = {s: dict(Counter(r['class_name'] for r in resolved if r['split'] == s)) for s in groups}
        all_classes = {r['class_name'] for r in rows}
        missing_classes = {s: sorted(all_classes - set(coverage[s])) for s in groups}
        summary = dict(dataset=dataset, input_sha256=digest(manifest), total_rows=len(rows),
                       recovered=len(resolved), unresolved=len(unresolved), original_content_overlap=overlaps,
                       split_counts=dict(Counter(r['split'] for r in resolved)), class_counts=coverage,
                       missing_classes=missing_classes, seed=2026,
                       split_unit='SHA256 of original image bytes; 80/10/10 hash assignment',
                       usable_for_existing_checkpoint_validation=False,
                       prospective_split_ready=not unresolved and overlaps == 0 and not any(missing_classes.values()),
                       limitation='Original-photo grouping does not guarantee structure/site independence. Mechanism labels are morphological proxies, requiring expert annotation.',
                       next_training_requirement='Train new checkpoint lineage using only research_v1/train.csv, including transfer-learning ancestors; old weights have held-out exposure.')
        (out / 'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        outputs.append(summary)
        print(json.dumps(summary), flush=True)
    return outputs


def inventory():
    """Capture machine-readable evidence without importing GPU frameworks."""
    experiments = ['unconditional_ddpm', 'conditional_ddpm', 'conditional_ddpm_128_augmented',
                   'crack_type_conditional', 'structural_mechanism_ddpm', 'exp6_level_c_spatial_physics']
    entries = []
    for n, name in enumerate(experiments, 1):
        folder = ROOT / 'experiments' / name
        corrected = folder / 'evaluation_corrected/EVALUATION_METRICS.json'
        metrics = corrected if corrected.exists() else folder / 'EVALUATION_METRICS.json'
        data = json.loads(metrics.read_text(encoding='utf-8')) if metrics.exists() else {}
        entries.append(dict(experiment=n, directory=name, metrics_path=metrics.relative_to(ROOT).as_posix(),
                            metrics_sha256=digest(metrics) if metrics.exists() else None,
                            score_kind='heuristic agreement; requires detector validation',
                            corrected_evaluation=metrics == corrected,
                            confusion=data.get('confusion_matrix', {}),
                            fidelity=data.get('fidelity'),
                            plot=f'experiments/{name}/plots/exp{n}_overview.png'))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    snapshot = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(), commit=commit, worktree_may_have_uncommitted_changes=True,
                    packages={p.metadata['Name']: p.version for p in importlib.metadata.distributions()},
                    script_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                                   [Path(__file__), ROOT / 'src/evaluate_exp6_spatial_physics_256.py', ROOT / 'src/plot_all_experiments.py']},
                    experiments=entries,
                    publication_readiness='incomplete',
                    remaining_gates=['Expert-validated mechanism labels and detector calibration',
                        'New clean training lineage with original-photo grouped splits',
                        'Matched-sample fidelity with identical preprocessing and independent references',
                        'Multiple-seed runs and uncertainty analysis',
                        'Spatial-conditioning ablations and held-out external-dataset evaluation'])
    out = ROOT / 'experiments/research_inventory.json'
    out.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(f'Saved {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'inventory'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else inventory()
