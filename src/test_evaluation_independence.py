"""Regression tests that do not load torch or model weights."""
import ast
from pathlib import Path
import unittest


class PredictionIndependence(unittest.TestCase):
    def test_classifier_ignores_target_and_commanded_values(self):
        source = Path(__file__).with_name('evaluate_exp6_spatial_physics_256.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_mechanism_class')
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<classifier>', 'exec'), namespace)
        features = dict(has_crack=True, n_sig=1, c_ratio=.95, max_diag=150.,
                        n_junction_pixels=0, parallel_diag=0, delta_theta=0., taper_ratio=1., theta=90.)
        predict = namespace['predict_mechanism_class']
        self.assertEqual(predict(features), 1)
        for requested in range(7):
            self.assertEqual(predict(dict(features, class_id=requested, cmd_theta=requested*20)), 1)
        self.assertEqual(predict(dict(features, has_crack=False)), 0)
        self.assertIn('pred_c = s["pred_class"]', source)
        self.assertNotIn('pred_c = cid if adhered', source)


if __name__ == '__main__':
    unittest.main()
