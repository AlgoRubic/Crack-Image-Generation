"""Guard comparison failure behavior; no expensive image evaluation in unit tests."""
import unittest
from reproduce_results import compare


class ComparisonTests(unittest.TestCase):
    def test_missing_numeric_field_fails(self):
        count, differences = compare({'classes': {'cracked': {'samples': 200}}}, {})
        self.assertEqual(count, 1)
        self.assertEqual(differences[0]['field'], 'classes.cracked.samples')

    def test_matrix_change_fails(self):
        count, differences = compare({'matrix': [[10, 0], [1, 9]]}, {'matrix': [[10, 0], [0, 10]]})
        self.assertEqual(count, 4)
        self.assertEqual(len(differences), 2)

    def test_explicit_fidelity_exclusion_and_tolerance(self):
        count, differences = compare({'fidelity': {'fid': 10}, 'rate': .4},
                                     {'fidelity': {'status': 'not_requested'}, 'rate': .40001})
        self.assertEqual(count, 1)
        self.assertFalse(differences)


if __name__ == '__main__':
    unittest.main()
