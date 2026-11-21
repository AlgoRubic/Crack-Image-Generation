import importlib.util
from pathlib import Path
import unittest
import csv
import json

spec = importlib.util.spec_from_file_location('research', Path(__file__).with_name('research.py'))
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


class SplitTests(unittest.TestCase):
    def test_same_original_cannot_cross_splits(self):
        for group in ['original-a', 'original-b', 'identical-content-sha']:
            splits = {research.assign_split(group, 2026) for _ in range(20)}
            self.assertEqual(len(splits), 1)

    def test_reproducible_and_all_partitions_populated(self):
        first = [research.assign_split(str(i), 2026) for i in range(1000)]
        self.assertEqual(first, [research.assign_split(str(i), 2026) for i in range(1000)])
        self.assertEqual(set(first), {'train', 'validation', 'test'})
        self.assertNotEqual(first, [research.assign_split(str(i), 42) for i in range(1000)])

    def test_actual_splits_and_transfer_ancestor_separation(self):
        assignments = {}
        for dataset in ('exp4_level_a_morphology_256', 'exp5_exp6_level_bc_mechanisms_256'):
            folder = research.ROOT / 'datasets' / dataset / 'research_v1'
            audit_path = folder / 'audit.json'
            if not audit_path.exists():
                self.skipTest('Run python pipeline/research.py prepare first')
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit['unresolved'], 0)
            self.assertTrue(audit['prospective_split_ready'])
            count = 0
            for split in ('train', 'validation', 'test'):
                with (folder / (split + '.csv')).open(encoding='utf-8') as stream:
                    for row in csv.DictReader(stream):
                        group = row['original_sha256']
                        self.assertEqual(assignments.setdefault(group, split), split,
                                         'Original-image leakage across split or transfer ancestor')
                        self.assertEqual(research.assign_split(group, 2026), split)
                        self.assertTrue((research.ROOT / row['path']).is_file())
                        self.assertTrue((research.ROOT / row['original_path']).is_file())
                        count += 1
            self.assertEqual(count, audit['total_rows'])


if __name__ == '__main__':
    unittest.main()
