import unittest

from safety_vision.geometry import denormalize_polygon, normalize_polygon, point_in_polygon


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_inside(self) -> None:
        self.assertTrue(point_in_polygon((5, 5), self.square))

    def test_outside(self) -> None:
        self.assertFalse(point_in_polygon((15, 5), self.square))

    def test_boundary_counts_as_inside(self) -> None:
        self.assertTrue(point_in_polygon((10, 5), self.square))

    def test_normalization_round_trip(self) -> None:
        points = [(20, 10), (100, 50), (180, 90)]
        normalized = normalize_polygon(points, 200, 100)
        self.assertEqual(denormalize_polygon(normalized, 200, 100), points)


if __name__ == "__main__":
    unittest.main()

