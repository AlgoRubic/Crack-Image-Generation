"""Conservatively link all similarity candidates before assigning new splits.

This groups suspected aliases, including same-split pairs, without declaring
them confirmed duplicates. False positives can reduce independent group count.
"""
import csv
import hashlib
import json
from screen_near_duplicates import ROOT, candidate_pairs, load_assignments


def main():
    folder = ROOT / 'datasets/near_duplicate_screen_v1'
    evidence = json.loads((folder / 'research_v4_summary.json').read_text())
    index = folder / 'descriptors.csv'
    source = ROOT / 'datasets/source_content_v1/index.csv'
    if (evidence['status'] != 'completed' or
            hashlib.sha256(index.read_bytes()).hexdigest() != evidence['descriptors_sha256'] or
            hashlib.sha256(source.read_bytes()).hexdigest() != evidence['source_index_sha256']):
        raise ValueError('Completed source screening evidence changed')
    assignments, inputs = load_assignments('research_v4')
    if inputs != evidence['manifest_sha256']:
        raise ValueError('Screened split manifests changed')
    with index.open(encoding='utf-8', newline='') as stream:
        records = {r['original_path']: r for r in csv.DictReader(stream)}
    if set(records) != set(assignments):
        raise ValueError('Descriptor coverage differs from split originals')
    rows = [{**records[p], 'split': assignments[p][0]} for p in sorted(records)]
    links = [dict(path_a=a['original_path'], path_b=b['original_path'],
                  phash_distance=pd, dhash_distance=dd)
             for a, b, pd, dd in candidate_pairs(rows, cross_split_only=False)]
    path = folder / 'conservative_links.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path_a', 'path_b', 'phash_distance', 'dhash_distance'])
        writer.writeheader()
        writer.writerows(links)
    summary = dict(status='completed', pairs=len(links),
                   source_index_sha256=evidence['source_index_sha256'],
                   descriptors_sha256=evidence['descriptors_sha256'],
                   links_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   protocol=evidence['protocol'],
                   policy='Group ALL declared-threshold similarity candidates, including within-split pairs, before reassigning components. This is conservative isolation, not confirmed-duplicate labeling.',
                   caveat='False-positive links can reduce independent group count. Crops/rotations/larger edits and site identity remain unverified.')
    (folder / 'conservative_links_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
