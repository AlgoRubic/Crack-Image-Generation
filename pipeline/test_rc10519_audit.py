import unittest
from audit_rc10519 import shape_issues


class AnnotationGeometryTests(unittest.TestCase):
    def test_reversed_corners_are_valid_but_zero_area_is_flagged(self):
        self.assertEqual(shape_issues(dict(shape_type='rectangle',points=[[10,10],[0,0]]),10,10),[])
        self.assertIn('degenerate_rectangle',shape_issues(dict(shape_type='rectangle',points=[[0,0],[0,10]]),10,10))

    def test_bad_coordinates_are_reported_without_silent_clipping(self):
        self.assertIn('out_of_image_bounds',shape_issues(dict(shape_type='rectangle',points=[[-1,0],[5,5]]),10,10))
        self.assertIn('invalid_or_nonfinite_points',shape_issues(dict(points=[[float('nan'),0]]),10,10))
        self.assertIn('invalid_or_nonfinite_points',shape_issues(dict(points=[['1',0]]),10,10))


if __name__=='__main__': unittest.main()
