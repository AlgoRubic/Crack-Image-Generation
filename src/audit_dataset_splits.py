"""Read-only manifest audit. Produces evidence, never relabels trained data as held-out."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for folder in sorted((ROOT / 'datasets').iterdir()):
        if not folder.is_dir():
            continue
        manifests = {}
        for file in sorted(folder.glob('*.csv')):
            with file.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            manifests[file.name] = rows
        if not manifests:
            continue
        entry = {'dataset': folder.name, 'manifests': {}, 'overlap_checks': []}
        for name, rows in manifests.items():
            paths = [r.get('path', '') for r in rows]
            parents = [r.get('original_path', '') for r in rows]
            entry['manifests'][name] = {
                'rows': len(rows), 'unique_paths': len(set(paths)),
                'missing_images': sum(not (ROOT / p).is_file() for p in paths),
                'rows_with_original_path': sum(bool(p) for p in parents),
                'parent_group_split_possible': bool(rows) and all(parents),
            }
        names = list(manifests)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                check = {'a': a, 'b': b}
                for key in ('path', 'original_path'):
                    sa = {r.get(key) for r in manifests[a] if r.get(key)}
                    sb = {r.get(key) for r in manifests[b] if r.get(key)}
                    check[key + '_overlap'] = len(sa & sb) if sa and sb else None
                entry['overlap_checks'].append(check)
        entry['caveat'] = 'Disjoint paths alone do not establish independence. Existing trained checkpoints retain all original training exposure.'
        records.append(entry)
    out = ROOT / 'datasets' / 'SPLIT_AUDIT.json'
    out.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
