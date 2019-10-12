# To run tests:
# python -m unittest discover tests

import unittest
import recoengi
import recoengi.cf as cf
import numpy as np

class TestCF(unittest.TestCase):

    def test_hello(self):
        M = np.array([[1,2,3,4], [1, 1, 0, 0], [1, -3, 0, 0.5]])
        pm = cf.PM(M)
        pm.computeSimilarityMatrix(bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = 2)

        self.assertTrue(abs(pm.S.sum() - 3.6035) < 0.01)


if __name__ == '__main__':
    unittest.main()