"""Test paired-control construction without loading a checkpoint or running inference."""
import ast
from pathlib import Path
import unittest
import torch
import numpy as np
from summarize_ablation import geometry


class Controls(unittest.TestCase):
    def test_no_detection_is_explicit_not_a_fabricated_distance(self):
        command=np.zeros((2,16,16),dtype=np.float32);command[0,:,8]=1
        result=geometry(np.zeros((16,16),bool),command)
        self.assertTrue(result['no_detection'])
        self.assertEqual(result['recall'],0)
        self.assertIsNone(result['mae'])

    def test_perfect_trajectory_has_zero_distance(self):
        command=np.zeros((2,16,16),dtype=np.float32);command[0,:,8]=1
        result=geometry(command[0].astype(bool),command)
        self.assertEqual(result['recall'],1)
        self.assertEqual(result['mae'],0)
        self.assertFalse(result['no_detection'])

    def test_paired_controls_preserve_input_and_shuffle_within_class_batch(self):
        tree=ast.parse(Path(__file__).with_name('ablate_spatial_control.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variants')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<variants>','exec'),namespace)
        source=torch.arange(16,dtype=torch.float32).reshape(2,2,2,2)
        saved=source.clone()
        controls=namespace['variants'](source)
        self.assertTrue(torch.equal(controls['correct'],saved))
        self.assertFalse(controls['zero'].any())
        self.assertTrue(torch.equal(controls['shuffled'][0],saved[1]))
        self.assertTrue(torch.equal(controls['shuffled'][1],saved[0]))
        self.assertTrue(torch.equal(source,saved))
        with self.assertRaises(ValueError):
            namespace['variants'](source[:1])


if __name__=='__main__':
    unittest.main()
