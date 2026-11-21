import unittest
import numpy as np
from evaluate_research import geometry_scores,validate_command
from sample_research import validate_config


class ResearchHandoff(unittest.TestCase):
    def test_reference_commands_fail_closed(self):
        empty=np.zeros((2,256,256),np.float32)
        validate_command(empty,0)
        with self.assertRaises(ValueError):validate_command(empty,1)
        with self.assertRaises(ValueError):validate_command(np.zeros((256,256)),1)
    def test_smoke_requires_explicit_opt_in(self):
        config=dict(initialization='scratch',transfer_ancestors=[],smoke_only=True,experiment=6,
                    conditioning='spatial',split_version='research_v4')
        with self.assertRaises(ValueError):validate_config(config,False)
        validate_config(config,True)
        with self.assertRaises(ValueError):validate_config(dict(config,initialization='historical'),True)
        with self.assertRaises(ValueError):validate_config(dict(config,smoke_only=False),False)
        validate_config(dict(config,smoke_only=False,split_version='research_v5'),False)

    def test_noise_coverage_does_not_imply_good_trajectory_f1(self):
        command=np.zeros((2,64,64),np.float32);command[0,:,32]=1
        score=geometry_scores(np.ones((64,64),bool),command)
        self.assertEqual(score['trajectory_recall'],1)
        self.assertLess(score['trajectory_precision'],.2)
        self.assertLess(score['trajectory_f1'],.3)

    def test_empty_and_perfect_detections(self):
        command=np.zeros((2,64,64),np.float32);command[0,:,32]=1
        empty=geometry_scores(np.zeros((64,64),bool),command)
        self.assertEqual(empty['trajectory_f1'],0)
        self.assertIsNone(empty['conditional_distance_mae_px'])
        perfect=geometry_scores(command[0].astype(bool),command)
        self.assertEqual(perfect['trajectory_f1'],1)
        self.assertEqual(perfect['conditional_distance_mae_px'],0)


if __name__=='__main__':unittest.main()
