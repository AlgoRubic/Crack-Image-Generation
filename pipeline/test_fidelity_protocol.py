import random
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from fidelity_research import assert_disjoint,balanced_select
import fidelity_research


class FidelityProtocol(unittest.TestCase):
    def test_smoke_metrics_cannot_be_computed(self):
        args=SimpleNamespace(samples='unused',allow_smoke_preflight=True,compute=True)
        with patch.object(fidelity_research,'verify_samples',return_value=({'smoke_only':True},'hash')):
            with self.assertRaisesRegex(ValueError,'Smoke checkpoints'):
                fidelity_research.main(args)
    def test_each_leakage_identity_is_checked(self):
        train=[dict(acquisition_group='a',original_sha256='b',original_rgb_sha256='c')]
        clean=dict(acquisition_group='x',original_sha256='y',original_rgb_sha256='z')
        assert_disjoint(train,[clean])
        for key in clean:
            with self.assertRaises(ValueError):assert_disjoint(train,[dict(clean,**{key:train[0][key]})])

    def test_budget_cannot_silently_shrink(self):
        rows=[dict(class_id=0,path='one',acquisition_group='a')]
        with self.assertRaises(ValueError):balanced_select(rows,1,2,random.Random(2026))

    def test_class_balance_and_group_spreading_are_reproducible(self):
        rows=[dict(class_id=c,path=f'{c}-{i}',acquisition_group=f'{c}-{i//2}') for c in range(2) for i in range(6)]
        a=balanced_select(rows,2,3,random.Random(2026));b=balanced_select(rows,2,3,random.Random(2026))
        self.assertEqual(a,b)
        for cid in range(2):
            selected=[r for r in a if r['class_id']==cid]
            self.assertEqual(len(selected),3)
            self.assertEqual(len({r['acquisition_group'] for r in selected}),3)


if __name__=='__main__':unittest.main()
