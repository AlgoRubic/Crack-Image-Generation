"""Limited, reproducible cross-split perceptual similarity screening.

Candidates require pHash distance <=3 AND dHash distance <=6. Four exact
16-bit pHash bands enumerate every pair within the pHash threshold. This is
not a crop/rotation invariant search or a declaration of data independence.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from research import ROOT

DATASETS = ('exp1_exp2_sdnet_64', 'exp3_augmented_128',
            'exp4_level_a_morphology_256', 'exp5_exp6_level_bc_mechanisms_256')
FIELDS = ['original_path', 'sha256', 'decoded_rgb_sha256', 'size_bytes',
          'mtime_ns', 'phash', 'dhash']


def bits_hex(bits):
    value = 0
    for bit in np.asarray(bits).ravel():
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def descriptors(image):
    gray = image.convert('L')
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(small, type=2, norm='ortho', workers=1)[:8, :8]
    threshold = np.median(low.ravel()[1:])
    phash_bits = low > threshold
    phash_bits[0, 0] = False  # Brightness/DC is deliberately excluded.
    wide = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
    return bits_hex(phash_bits), bits_hex(wide[:, 1:] > wide[:, :-1])


def describe(record, previous):
    path = ROOT / record['original_path']
    stat = path.stat()
    if (stat.st_size != int(record['size_bytes']) or
            stat.st_mtime_ns != int(record['mtime_ns'])):
        raise ValueError(f'Source changed since exact-content audit: {path}')
    prior = previous.get(record['original_path'])
    if prior and all(str(prior[k]) == str(record[k]) for k in FIELDS[:5]):
        return prior, False
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError(f'Source byte hash mismatch: {path}')
    with Image.open(io.BytesIO(content)) as image:
        rgb = image.convert('RGB')
        prefix = f'{rgb.width}x{rgb.height}:RGB:'.encode()
        if hashlib.sha256(prefix + rgb.tobytes()).hexdigest() != record['decoded_rgb_sha256']:
            raise ValueError(f'Source RGB hash mismatch: {path}')
        phash, dhash = descriptors(rgb)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError(f'Source changed during screening: {path}')
    return {**{k: record[k] for k in FIELDS[:5]}, 'phash': phash, 'dhash': dhash}, True


def candidate_pairs(records, cross_split_only=True):
    """Yield unique cross-split pairs; exact pixels already covered elsewhere."""
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        phash, dhash = int(record['phash'], 16), int(record['dhash'], 16)
        previous = set()
        for band in range(4):
            previous.update(buckets[(band, (phash >> (16 * band)) & 65535)])
        for j in sorted(previous):
            other = records[j]
            if cross_split_only and record['split'] == other['split']:
                continue
            if record['decoded_rgb_sha256'] == other['decoded_rgb_sha256']:
                continue
            pd = (phash ^ int(other['phash'], 16)).bit_count()
            if pd > 3:
                continue
            dd = (dhash ^ int(other['dhash'], 16)).bit_count()
            if dd <= 6:
                yield other, record, pd, dd
        for band in range(4):
            buckets[(band, (phash >> (16 * band)) & 65535)].append(i)


def load_assignments(version):
    assignments, inputs = {}, {}
    for dataset in DATASETS:
        for split in ('train', 'validation', 'test'):
            path = ROOT / 'datasets' / dataset / version / f'{split}.csv'
            inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    value = (split, row['acquisition_group'])
                    prior = assignments.setdefault(row['original_path'], value)
                    if prior != value:
                        raise ValueError(f'Inconsistent cross-experiment split: {row["original_path"]}')
    return assignments, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-version', default='research_v4')
    args = parser.parse_args()
    assignments, inputs = load_assignments(args.split_version)
    source = ROOT / 'datasets/source_content_v1/index.csv'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = json.loads(source.with_name('summary.json').read_text())
    if expected['status'] != 'completed' or source_sha != expected['index_sha256']:
        raise ValueError('Exact-content source index is incomplete or changed')
    with source.open(encoding='utf-8', newline='') as stream:
        sources = {r['original_path']: r for r in csv.DictReader(stream)}
    if set(sources) != set(assignments):
        raise ValueError('Manifest originals differ from audited source index')
    out = ROOT / 'datasets' / 'near_duplicate_screen_v1'
    out.mkdir(exist_ok=True)
    index = out / 'descriptors.csv'
    previous = {}
    if index.exists():
        with index.open(encoding='utf-8', newline='') as stream:
            previous = {r['original_path']: r for r in csv.DictReader(stream)}
    records = []
    with index.open('a', encoding='utf-8', newline='') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for i, (record, changed) in enumerate(pool.map(lambda r: describe(r, previous),
                                                     [sources[k] for k in sorted(sources)]), 1):
            if changed:
                writer.writerow(record)
            split, group = assignments[record['original_path']]
            records.append({**record, 'split': split, 'acquisition_group': group})
            if i % 1000 == 0:
                stream.flush()
                print(f'Descriptors verified {i}/{len(sources)}', flush=True)
    rows = []
    for a, b, pd, dd in candidate_pairs(records):
        rows.append(dict(path_a=a['original_path'], path_b=b['original_path'],
                         split_a=a['split'], split_b=b['split'],
                         group_a=a['acquisition_group'], group_b=b['acquisition_group'],
                         phash_distance=pd, dhash_distance=dd,
                         review_decision='', reviewer='', notes=''))
    rows.sort(key=lambda r: (r['phash_distance'], r['dhash_distance'], r['path_a'], r['path_b']))
    candidates = out / f'{args.split_version}_candidates.csv'
    fields = ['path_a', 'path_b', 'split_a', 'split_b', 'group_a', 'group_b',
              'phash_distance', 'dhash_distance', 'review_decision', 'reviewer', 'notes']
    if candidates.exists():
        with candidates.open(encoding='utf-8', newline='') as stream:
            if any(r.get('review_decision') or r.get('reviewer') or r.get('notes') for r in csv.DictReader(stream)):
                raise ValueError('Refusing to overwrite populated review decisions')
    with candidates.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
                   split_version=args.split_version, source_files=len(records),
                   source_index_sha256=source_sha, manifest_sha256=inputs,
                   descriptors_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                   candidate_csv_sha256=hashlib.sha256(candidates.read_bytes()).hexdigest(),
                   cross_split_candidate_pairs=len(rows),
                   split_pair_counts=dict(Counter('+'.join(sorted((r['split_a'], r['split_b']))) for r in rows)),
                   protocol=dict(phash='32x32 grayscale LANCZOS; orthonormal DCT-II 8x8; median excluding DC; DC bit zero',
                                 dhash='9x8 grayscale LANCZOS; horizontal adjacent comparisons',
                                 thresholds=dict(phash_hamming_max=3, dhash_hamming_max=6),
                                 enumeration='Four 16-bit pHash bands; exhaustive within declared pHash threshold'),
                   interpretation='Unreviewed candidates only; no split or label changes applied.',
                   limitations=['Not calibrated on this dataset', 'Misses crops, rotations, flips and larger edits',
                                'Does not establish site/structure independence',
                                'Exact decoded duplicates excluded because checked by research_v4 content audit'])
    (out / f'{args.split_version}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
