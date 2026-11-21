"""Regression checks for shared detector and pixel-metric calculations."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.ndimage import uniform_filter, binary_opening
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crack_segmentation import fissure_mask
from validate_detector import metrics
from calibrate_detector import select


class DetectorTests(unittest.TestCase):
    def test_exact_historical_detector_equivalence(self):
        rgb = np.random.default_rng(2026).integers(0,256,(256,256,3),dtype=np.uint8)
        arr = rgb.astype(np.float32)
        gray = .299*arr[:,:,0]+.587*arr[:,:,1]+.114*arr[:,:,2]
        expected = binary_opening(uniform_filter(gray,size=23)-gray >24.5, structure=np.ones((2,2),bool))
        np.testing.assert_array_equal(fissure_mask(rgb),expected)

    def test_pixel_metrics_and_empty_denominators(self):
        result = metrics(3,1,2)
        self.assertEqual(result['precision'],.75)
        self.assertEqual(result['recall'],.6)
        self.assertEqual(result['iou'],.5)
        self.assertEqual(result['dice'],2/3)
        self.assertIsNone(metrics(0,0,0)['iou'])

    def test_clean_surface(self):
        self.assertFalse(fissure_mask(np.full((256,256,3),128,np.uint8)).any())

    def test_validation_selection_and_tie_break(self):
        candidates = [{'config':{'threshold':12},'iou':.3},
                      {'config':{'threshold':18},'iou':.4},
                      {'config':{'threshold':24.5},'iou':.4}]
        self.assertEqual(select(candidates)['config']['threshold'],18)
        self.assertEqual(len(candidates),3)

    def test_invalid_parameters_rejected(self):
        image=np.zeros((16,16,3),np.uint8)
        for config in ({'window':2},{'threshold':-1},{'opening':0}):
            with self.assertRaises(ValueError):
                fissure_mask(image,**config)


if __name__ == '__main__':
    unittest.main()
