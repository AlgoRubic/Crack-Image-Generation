"""Verify saved-analysis records and repeatability; requires local input images/masks."""
import argparse
import csv
import json
import sys
from pathlib import Path

from reproduce_results import ROOT, NAMES, EXPECTED, compare, sha, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-ids', nargs='+', required=True)
    parser.add_argument('--summary', type=Path, default=ROOT / 'experiments/saved_analysis_reproduction.json')
    args = parser.parse_args()
    records = []
    for number, name in enumerate(NAMES, 1):
        previous = None
        for run_id in args.run_ids:
            path = ROOT / 'experiments' / name / 'reproductions' / run_id / 'run_manifest.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['status'] != 'matched' or record['image_count'] != EXPECTED[number - 1]:
                raise ValueError(f'Incomplete or differing run: {relative(path)}')
            for filename, digest in record['input_sha256'].items():
                if sha(ROOT / filename) != digest:
                    raise ValueError(f'Input changed: {filename}')
            for filename, digest in record['output_sha256'].items():
                if sha(path.parent / filename) != digest:
                    raise ValueError(f'Output changed: {relative(path.parent / filename)}')
            if sha(ROOT / record['baseline']) != record['baseline_sha256']:
                raise ValueError(f'Baseline changed: {record["baseline"]}')
            with (path.parent / 'PER_SAMPLE_ERROR_LOG.csv').open(encoding='utf-8', newline='') as stream:
                sample_count = sum(1 for _ in csv.DictReader(stream))
            if sample_count != EXPECTED[number - 1]:
                raise ValueError(f'Wrong per-sample row count: {relative(path)}')
            current = json.loads((path.parent / 'EVALUATION_METRICS.json').read_text(encoding='utf-8'))
            baseline = json.loads((path.parent / 'baseline_metrics.json').read_text(encoding='utf-8'))
            checked, differences = compare(baseline, current)
            if not checked or differences:
                raise ValueError(f'Baseline numeric comparison failed: {relative(path)}')
            if previous is not None and current != previous:
                raise ValueError(f'Repeated metric JSON differs for Experiment {number}')
            previous = current
            records.append(dict(experiment=number, run_id=run_id, manifest=relative(path),
                                manifest_sha256=sha(path), images=sample_count,
                                numeric_fields_compared=checked, matched=True,
                                input_files_verified=len(record['input_sha256'])))
    summary = dict(status='passed', experiments=6, images_per_run=sum(EXPECTED),
                   run_ids=args.run_ids, repeated_metric_json_identical=len(args.run_ids) > 1,
                   scope='Saved-image morphology/spatial reanalysis; fidelity excluded; same local environment',
                   limitation='Not a claim of new generation, exact historical sampling, independent validity or cross-platform identity',
                   records=records)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Passed: six experiments, {sum(EXPECTED)} images per run, {len(args.run_ids)} run(s).')


if __name__ == '__main__':
    main()
