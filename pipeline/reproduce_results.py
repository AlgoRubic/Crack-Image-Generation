"""Reanalyse saved images in new experiment-local folders; never train or replace baselines.

Run from any directory: python pipeline/reproduce_results.py --run-id review-001
The default excludes fidelity, whose historical reference protocols differ across experiments.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['unconditional_ddpm', 'conditional_ddpm', 'conditional_ddpm_128_augmented',
         'crack_type_conditional', 'structural_mechanism_ddpm', 'exp6_level_c_spatial_physics']
SOURCES = ['evaluate_exp1_2_3_errors'] * 3 + ['evaluate_crack_type_256',
           'evaluate_exp5_mechanisms_256', 'evaluate_exp6_spatial_physics_256']
EXPECTED = [1000, 2000, 400, 80, 112, 1400]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def inputs(number):
    """Return exact image and reference inputs, including deterministic reference subsampling."""
    import numpy as np
    base = ROOT / 'experiments' / NAMES[number - 1]
    refs = []
    if number == 1:
        images = sorted((base / 'outputs/evaluation/unconditional_epoch_050').glob('*.png'))
    elif number in (2, 3):
        prefix = 'conditional_' if number == 2 else 'conditional_128_augmented_'
        nc = sorted((base / 'outputs/evaluation' / (prefix + 'non_cracked')).glob('*.png'))
        cr = sorted((base / 'outputs/evaluation' / (prefix + 'cracked')).glob('*.png'))
        images = nc + cr
        if len(nc) != EXPECTED[number - 1] // 2 or len(cr) != EXPECTED[number - 1] // 2:
            raise ValueError(f'Exp {number}: incomplete class image set ({len(nc)}, {len(cr)})')
    else:
        folder = base / ('results' if number == 6 else 'generated_samples')
        images = sorted(folder.glob('*/sample_*.png'))
        classes = (['non_cracked', 'longitudinal', 'transverse', 'diagonal', 'alligator_map'] if number == 4
                   else ['non_cracked', 'flexural', 'flexure_shear', 'web_shear', 'torsion',
                         'corrosion_splitting', 'shrinkage_map'])
        per_class = 200 if number == 6 else 16
        observed = {name: sum(p.parent.name == name for p in images) for name in classes}
        if any(value != per_class for value in observed.values()):
            raise ValueError(f'Exp {number}: incomplete class image sets: {observed}')
    if len(images) != EXPECTED[number - 1]:
        raise ValueError(f'Exp {number}: expected {EXPECTED[number - 1]} images, found {len(images)}')
    if number <= 3:
        manifest = ROOT / ('data/processed_128/train.csv' if number == 3 else 'data/processed/train.csv')
        with manifest.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        rng = np.random.default_rng(2026)
        refs.append(manifest)
        if number == 1:
            refs += [ROOT / rows[i]['path'] for i in rng.choice(len(rows), size=min(300, len(images)), replace=False)]
        else:
            for label, count in [('non_cracked', len(nc)), ('cracked', len(cr))]:
                candidates = [ROOT / row['path'] for row in rows if row['label'] == label]
                refs += [candidates[i] for i in rng.choice(len(candidates), size=min(200, count, len(candidates)), replace=False)]
    if number == 6:
        for image in images:
            refs.append(image.parent.parent / (image.parent.name + '_physics_masks') /
                        ('phys_' + image.stem.removeprefix('sample_') + '.npy'))
    paths = images + refs
    missing = [relative(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Exp {number}: missing inputs: {missing[:10]}')
    return paths, len(images)


def compare(baseline, current, prefix=''):
    """Compare numeric observations only; exclusions are recorded in the run manifest."""
    checked, differences = 0, []
    if isinstance(baseline, dict):
        for key, value in baseline.items():
            if key in {'fidelity', 'fid_kid', 'baseline_comparison'}:
                continue
            n, diff = compare(value, current.get(key) if isinstance(current, dict) else None,
                              f'{prefix}.{key}'.strip('.'))
            checked += n
            differences.extend(diff)
    elif isinstance(baseline, list):
        for index, value in enumerate(baseline):
            other = current[index] if isinstance(current, list) and index < len(current) else None
            n, diff = compare(value, other, f'{prefix}[{index}]')
            checked += n
            differences.extend(diff)
    elif isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        checked = 1
        equal = isinstance(current, (int, float)) and (
            (math.isnan(baseline) and math.isnan(current)) or
            math.isclose(baseline, current, rel_tol=0, abs_tol=1e-4))
        if not equal:
            differences.append(dict(field=prefix, baseline=baseline, reproduced=current))
    return checked, differences


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='New folder name; existing runs are never overwritten')
    parser.add_argument('--experiments', nargs='+', type=int, choices=range(1, 7), default=list(range(1, 7)))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run_id):
        parser.error('run-id must contain only letters, digits, underscore or hyphen, starting with a letter/digit')
    for variable, folder in [('TORCH_HOME', '.cache/torch'), ('MPLCONFIGDIR', '.cache/matplotlib'),
                             ('TEMP', '.cache/tmp'), ('TMP', '.cache/tmp')]:
        target = ROOT / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import numpy as np
    import scipy
    import PIL
    torch.set_num_threads(2)
    plans = []
    # Complete preflight before creating any result folders.
    for number in sorted(set(args.experiments)):
        base = ROOT / 'experiments' / NAMES[number - 1]
        out = base / 'reproductions' / args.run_id
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite {out}')
        baseline = base / ('evaluation_corrected/EVALUATION_METRICS.json' if number == 6 else 'EVALUATION_METRICS.json')
        baseline.read_bytes()
        paths, count = inputs(number)
        plans.append((number, base, out, baseline, paths, count))
    successes = True
    for number, base, out, baseline, paths, count in plans:
        out.mkdir(parents=True)
        source = ROOT / 'src' / (SOURCES[number - 1] + '.py')
        dependencies = [source, Path(__file__).resolve()]
        if number == 5:
            dependencies.append(ROOT / 'src/prepare_experiment5_mechanisms.py')
        if number == 6:
            dependencies.append(ROOT / 'src/crack_segmentation.py')
        hashed = {relative(p): sha(p) for p in sorted(set(paths + dependencies))}
        before = sha(baseline)
        record = dict(experiment=number, run_id=args.run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Saved-image morphology and spatial analysis; no retraining, generation or fidelity',
                      image_count=count, input_sha256=hashed, baseline=relative(baseline), baseline_sha256=before,
                      environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                       numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__, device='cpu'),
                      comparison_policy=dict(absolute_tolerance=1e-4, relative_tolerance=0,
                                             excluded=['fidelity', 'fid_kid', 'baseline_comparison'],
                                             scope='All other numeric fields; text metadata not compared'), status='running')
        record['environment']['packages'] = {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'numpy', 'scipy', 'pillow', 'matplotlib', 'torch-fidelity', 'tqdm')}
        (out / 'baseline_metrics.json').write_bytes(baseline.read_bytes())
        write_json(out / 'run_manifest.json', record)
        print(f'Experiment {number}: analysing {count} saved images -> {relative(out)}', flush=True)
        try:
            with (out / 'evaluation.log').open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if number == 6:
                        subprocess.run([sys.executable, str(source), '--output-dir', str(out),
                                        '--samples-dir', str(base / 'results'), '--figure-note',
                                        'Saved-image reproduction; fidelity not requested'], cwd=ROOT,
                                       stdout=log, stderr=log, check=True)
                    else:
                        module = importlib.import_module(SOURCES[number - 1])
                        if number == 1:
                            module.evaluate_experiment_1(torch.device('cpu'), out, with_fidelity=False)
                        elif number in (2, 3):
                            prefix = 'conditional_' if number == 2 else 'conditional_128_augmented_'
                            module.evaluate_binary_conditional_experiment(number, out,
                                base / 'outputs/evaluation' / (prefix + 'non_cracked'),
                                base / 'outputs/evaluation' / (prefix + 'cracked'),
                                ROOT / ('data/processed_128/train.csv' if number == 3 else 'data/processed/train.csv'),
                                128 if number == 3 else 64, torch.device('cpu'), with_fidelity=False)
                        elif number == 4:
                            module.main(output_dir=out)
                        else:
                            module.main(output_dir=out, with_fidelity=False)
            current = json.loads((out / 'EVALUATION_METRICS.json').read_text(encoding='utf-8'))
            checked, differences = compare(json.loads(baseline.read_text(encoding='utf-8')), current)
            unchanged = before == sha(baseline)
            input_unchanged = all(sha(ROOT / name) == digest for name, digest in hashed.items())
            record.update(status='matched' if checked and not differences and unchanged and input_unchanged else 'differences_found',
                          numeric_fields_compared=checked, differences=differences,
                          baseline_unchanged=unchanged, inputs_unchanged=input_unchanged,
                          output_sha256={p.name: sha(p) for p in sorted(out.iterdir())
                                         if p.is_file() and p.name not in {'run_manifest.json', 'evaluation.log'}})
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(out / 'run_manifest.json', record)
        print(f'Experiment {number}: {record["status"]}', flush=True)
        successes &= record['status'] == 'matched'
    if not successes:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
