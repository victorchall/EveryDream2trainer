import unittest

from utils.first_fit_decreasing import first_fit_decreasing

class TestFirstFitDecreasing(unittest.TestCase):

    def test_single_basic(self):
        input = [[1, 2, 3, 4, 5, 6]]
        output = first_fit_decreasing(input, batch_size=2)
        self.assertEqual(output, [1, 2, 3, 4, 5, 6])

        input = [[1, 2, 3, 4, 5, 6]]
        output = first_fit_decreasing(input, batch_size=3)
        self.assertEqual(output, [1, 2, 3, 4, 5, 6])

        input = [[1, 2, 3, 4, 5, 6]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [1, 2, 3, 4, 5, 6])

        input = [[1, 2, 3]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [1, 2, 3])

    def test_multi_basic(self):
        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=2)
        self.assertEqual(output, [1, 1, 1, 1, 2, 2])

        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=3)
        self.assertEqual(output, [1, 1, 1, 2, 2, 1])

        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [1, 1, 1, 1, 2, 2])

        input = [[1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [2, 2, 1, 1])

    def test_multi_complex(self):
        input = [[1, 1, 1, 1], [2, 2], [3, 3, 3]]
        output = first_fit_decreasing(input, batch_size=2)
        self.assertEqual(output, [1, 1, 3, 3, 1, 1, 2, 2, 3])

        input = [[1, 1, 1, 1], [2, 2], [3, 3, 3]]
        output = first_fit_decreasing(input, batch_size=3)
        self.assertEqual(output, [1, 1, 1, 3, 3, 3, 2, 2, 1])

        input = [[1, 1, 1, 1], [2, 2], [3, 3, 3]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [1, 1, 1, 1, 3, 3, 3, 2, 2])

        input = [[1, 1], [2, 2], [3, 3, 3]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [3, 3, 3, 2, 1, 1, 2])

        input = [[1, 1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5]]
        output = first_fit_decreasing(input, batch_size=4)
        self.assertEqual(output, [1, 1, 1, 1, 4, 4, 4, 3, 2, 2, 2, 3, 5, 5, 3])

    def test_filler_bucket(self):
        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=2, filler_items=[9, 9])
        self.assertEqual(output, [1, 1, 1, 1, 2, 2, 9, 9])

        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=3, filler_items=[9, 9])
        self.assertEqual(output, [1, 1, 1, 2, 2, 9, 1, 9])

        input = [[1, 1, 1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=4, filler_items=[9, 9])
        self.assertEqual(output, [1, 1, 1, 1, 2, 2, 9, 9])

        input = [[1, 1], [2, 2]]
        output = first_fit_decreasing(input, batch_size=4, filler_items=[9, 9])
        self.assertEqual(output, [2, 2, 9, 9, 1, 1])

        input = [[1, 1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5]]
        output = first_fit_decreasing(input, batch_size=4, filler_items=[9, 9])
        self.assertEqual(output, [1, 1, 1, 1, 4, 4, 4, 9, 3, 3, 3, 9, 2, 2, 2, 5, 5])
