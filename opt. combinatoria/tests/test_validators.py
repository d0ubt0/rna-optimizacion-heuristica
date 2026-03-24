import unittest

import numpy as np

from tsp_mexico.src.io_utils import validate_matrix


class ValidatorTests(unittest.TestCase):
    def test_validate_matrix_rejects_non_square(self):
        with self.assertRaises(ValueError):
            validate_matrix(np.zeros((2, 3)), 'm')

    def test_validate_matrix_rejects_negative(self):
        matrix = np.array([[0.0, -1.0], [2.0, 0.0]])
        with self.assertRaises(ValueError):
            validate_matrix(matrix, 'm')


if __name__ == '__main__':
    unittest.main()
