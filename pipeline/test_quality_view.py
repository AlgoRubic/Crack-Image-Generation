"""Verify actual qualified data is an exact partition with unchanged assignments."""
import csv
import hashlib
import json
import unittest
from prepare_quality_view import ROOT, DATASET, VIEW


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class QualityViewTests(unittest.TestCase):
    def test_repository_view_is_complete_traceable_and_unchanged(self):
        base=ROOT/'datasets'/DATASET/'research_v5'
        view=base/VIEW
        audit=json.loads((view/'audit.json').read_text())
        inventory={r['image']:r for r in read(ROOT/'datasets/source_provenance/rc10519_annotation_inventory.csv')}
        exclusions=read(view/'excluded.csv')
        total=0
        for split in ('train','validation','test'):
            original={r['path']:r for r in read(base/(split+'.csv'))}
            selected=read(view/(split+'.csv'))
            excluded=[r for r in exclusions if r['split']==split]
            self.assertEqual(len(selected)+len(excluded),len(original))
            self.assertEqual({r['path'] for r in selected}|{r['path'] for r in excluded},set(original))
            self.assertFalse({r['path'] for r in selected}&{r['path'] for r in excluded})
            self.assertEqual(len({r['class_id'] for r in selected}),7)
            for row in selected:
                self.assertEqual(row,original[row['path']])
                if row['source']=='RC_10519':self.assertFalse(inventory[row['original_path']]['issue_flags'])
            for row in excluded:
                self.assertEqual(row['source'],'RC_10519')
                self.assertEqual(row['quality_exclusion_reason'],inventory[row['original_path']]['issue_flags'])
                self.assertTrue(row['quality_exclusion_reason'])
            self.assertEqual(hashlib.sha256((view/(split+'.csv')).read_bytes()).hexdigest(),audit['qualified_manifest_sha256'][split])
            total+=len(excluded)
        self.assertEqual(total,audit['excluded_total'])


if __name__=='__main__':unittest.main()
