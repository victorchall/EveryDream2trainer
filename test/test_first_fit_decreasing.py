import unittest

from utils.first_fit_decreasing import first_fit_decreasing

class TestFirstFitDecreasing(unittest.TestCase):

    def test_basic(self):
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
