"""Create a conservative geometry-qualified view of the unchanged v5 splits."""
import csv
import hashlib
import json
from collections import Counter
from research import ROOT, write_csv

DATASET = 'exp5_exp6_level_bc_mechanisms_256'
VIEW = 'strict_rc_geometry_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current_path = ROOT / 'datasets/current_research_split.json'
    current = json.loads(current_path.read_text())
    if current['version'] != 'research_v5':
        raise ValueError('This geometry qualification is defined for the frozen v5 split')
    inventory = ROOT / 'datasets/source_provenance/rc10519_annotation_inventory.csv'
    source_summary = json.loads(inventory.with_name('rc10519_local_inventory.json').read_text())
    if source_summary['status'] != 'completed' or sha(inventory) != source_summary['annotation_inventory_sha256']:
        raise ValueError('Annotation audit changed or is incomplete')
    with inventory.open(encoding='utf-8', newline='') as stream:
        records = {r['image']: r for r in csv.DictReader(stream)}
    base = ROOT / 'datasets' / DATASET / current['version']
    out = base / VIEW
    out.mkdir(exist_ok=False)
    excluded, counts, coverage, inputs = [], {}, {}, {}
    for split in ('train', 'validation', 'test'):
        file = base / (split + '.csv')
        inputs[file.relative_to(ROOT).as_posix()] = sha(file)
        with file.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        selected = []
        for row in rows:
            if row['source'] == 'RC_10519':
                if row['original_path'] not in records:
                    raise ValueError('RC row missing from annotation audit')
                flags = records[row['original_path']]['issue_flags']
                if flags:
                    excluded.append({**row, 'quality_exclusion_reason': flags})
                    continue
            selected.append(row)
        counts[split] = len(selected)
        coverage[split] = dict(Counter(r['class_name'] for r in selected))
        if len(coverage[split]) != 7:
            raise ValueError(f'Qualification loses a class in {split}')
        write_csv(out / (split + '.csv'), selected)
    write_csv(out / 'excluded.csv', excluded)
    report = dict(status='completed', dataset=DATASET, split_version='research_v5', quality_view=VIEW,
                  base_split_counts=current['datasets'][DATASET], qualified_split_counts=counts,
                  excluded_by_split=dict(Counter(r['split'] for r in excluded)), excluded_total=len(excluded),
                  excluded_issue_counts=dict(Counter(flag for r in excluded for flag in r['quality_exclusion_reason'].split(';'))),
                  class_counts=coverage, input_manifest_sha256=inputs,
                  annotation_inventory_sha256=sha(inventory),
                  qualified_manifest_sha256={s:sha(out/(s+'.csv')) for s in counts},
                  policy='Exclude any RC image with a flagged annotation geometry/metadata issue. Preserve v5 component IDs and split assignments; other source rows are unchanged.',
                  rationale='Exp5/6 proxy labels depend on RC bounding boxes. Exp3 uses image-level labels without those boxes and is not filtered by this view.',
                  limitations=['Conservative exclusion is not proof that every excluded label is wrong',
                               'Passing geometry checks does not validate crack presence, boxes, pixel masks, or mechanism proxy labels',
                               'Annotation dimensions are not automatically rescaled or repaired without source evidence'])
    (out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    current.setdefault('quality_views', {})[DATASET] = dict(name=VIEW, split_counts=counts,
                                                          audit_sha256=sha(out/'audit.json'))
    current_path.write_text(json.dumps(current, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
