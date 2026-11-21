"""Create/verify a portable manifest for final historical inference weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'models/checkpoint_manifest.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def create():
    entries=[]
    for folder in sorted((ROOT/'models/weights').iterdir()):
        if not folder.is_dir(): continue
        for path in sorted(folder.glob('*.pt')):
            entries.append(dict(experiment=int(folder.name[3]),path=path.relative_to(ROOT).as_posix(),
                                size_bytes=path.stat().st_size,sha256=sha(path),
                                lineage='historical_training_exposure; not research_v4-clean'))
    if len(entries)!=6 or {r['experiment'] for r in entries}!=set(range(1,7)):
        raise ValueError('Expected exactly one final inference checkpoint per experiment')
    MANIFEST.write_text(json.dumps(dict(schema_version=1,checkpoints=entries,
       distribution='Separate model bundle; weights are not in ordinary Git history'),indent=2),encoding='utf-8')
    print(f'Created {MANIFEST}; total {sum(r["size_bytes"] for r in entries)/2**20:.1f} MiB')


def verify():
    data=json.loads(MANIFEST.read_text());failures=[]
    for entry in data['checkpoints']:
        path=ROOT/entry['path']
        if not path.is_file(): failures.append(f'Missing Exp{entry["experiment"]} weight: {entry["path"]}')
        elif path.stat().st_size!=entry['size_bytes'] or sha(path)!=entry['sha256']:
            failures.append(f'Checksum mismatch: {entry["path"]}')
    print(json.dumps(dict(verified=not failures,checkpoints=len(data['checkpoints']),failures=failures),indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['create','verify'])
    create() if parser.parse_args().command=='create' else verify()
