import csv
from pathlib import Path
import unittest
from prepare_acquisition_splits import acquisition,ROOT


class AcquisitionTests(unittest.TestCase):
    def test_sibling_tiles_share_acquisition_across_labels(self):
        a=acquisition(dict(original_path='data/raw/SDNET2018/images/D/CD/7001-1.jpg'))
        b=acquisition(dict(original_path='data/raw/SDNET2018/images/D/UD/7001-100_2.jpg'))
        c=acquisition(dict(original_path='data/raw/SDNET2018/images/W/UW/7001-1.jpg'))
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)

    def test_malformed_tile_fails_closed(self):
        with self.assertRaises(ValueError):
            acquisition(dict(source='SDNET2018',original_path='data/raw/SDNET2018/images/D/CD/unknown.jpg'))

    def test_actual_cross_experiment_acquisition_and_content_separation(self):
        assignments={};content={};rgb={};source_groups={}
        for dataset in ('exp1_exp2_sdnet_64','exp3_augmented_128','exp4_level_a_morphology_256','exp5_exp6_level_bc_mechanisms_256'):
            folder=ROOT/'datasets'/dataset/'research_v4'
            for split in ('train','validation','test'):
                with (folder/(split+'.csv')).open(encoding='utf-8') as stream:
                    for row in csv.DictReader(stream):
                        self.assertEqual(assignments.setdefault(row['acquisition_group'],split),split)
                        self.assertEqual(source_groups.setdefault(row['source_acquisition_group'],split),split)
                        if row.get('original_sha256'):
                            self.assertEqual(content.setdefault(row['original_sha256'],split),split)
                        self.assertTrue(row.get('original_rgb_sha256'))
                        self.assertEqual(rgb.setdefault(row['original_rgb_sha256'],split),split)


if __name__=='__main__':
    unittest.main()
