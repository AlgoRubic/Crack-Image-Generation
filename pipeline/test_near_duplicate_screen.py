"""Check candidate coverage against brute force, including cross-split guards."""
import random
import unittest
from screen_near_duplicates import candidate_pairs


class ScreenTests(unittest.TestCase):
    def test_band_search_matches_brute_force(self):
        rng = random.Random(63)
        records = []
        for family in range(25):
            base = rng.getrandbits(63)
            for i in range(6):
                value = base
                for bit in rng.sample(range(63), i):
                    value ^= 1 << bit
                records.append(dict(phash=f'{value:016x}', dhash=f'{family:016x}',
                                    split=('train', 'validation', 'test')[i % 3],
                                    decoded_rgb_sha256=str(len(records)), id=len(records)))
        expected = {(a['id'], b['id']) for i, a in enumerate(records) for b in records[i+1:]
                    if a['split'] != b['split']
                    and (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count() <= 3
                    and (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count() <= 6}
        actual = [(a['id'], b['id']) for a, b, _, _ in candidate_pairs(records)]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_exact_pixels_excluded(self):
        rows = [dict(phash='0', dhash='0', split=s, decoded_rgb_sha256='same')
                for s in ('train', 'test')]
        self.assertEqual(list(candidate_pairs(rows)), [])

    def test_conservative_grouping_includes_same_split_links(self):
        rows=[dict(phash='0',dhash='0',split='train',decoded_rgb_sha256=str(i)) for i in range(3)]
        self.assertEqual(list(candidate_pairs(rows)),[])
        self.assertEqual(len(list(candidate_pairs(rows,cross_split_only=False))),3)


if __name__ == '__main__':
    unittest.main()
