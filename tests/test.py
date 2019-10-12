# To run tests:
# python -m unittest discover tests

import unittest
import recoengi
import recoengi.cf as cf
import numpy as np
from scipy import sparse

class TestCF(unittest.TestCase):

    def test_computeSimilarityMatrix(self):
        M = np.array([[1,2,3,4], [1, 1, 0, 0], [1, -3, 0, 0.5]])
        cfm = cf.CFM(M)
        cfm.computeSimilarityMatrix(bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = 2)
        self.assertTrue(abs(cfm.S.sum() - 3.6035) < 0.01)

    def test_computeEverythng(self):
        M = sparse.random(100, 10, density=0.3, format='csr')
        cfm = cf.CFM(M)
        cfm.computeEverything(bln_bin=False, bln_norm=True, flt_ths=0.0, ntop=10, flt_lb=-1)
        self.assertEqual(cfm.M.shape, (100, 10))
        self.assertEqual(cfm.B.shape, (100, 10))
        self.assertEqual(cfm.S.shape, (100, 100))
        self.assertEqual(cfm.SNORMALIZED.diagonal().max(), 0)
        self.assertEqual(cfm.SCORES.shape, (100, 10))
        self.assertEqual(cfm.AMOUNTS.shape, (100, 10))



if __name__ == '__main__':
    unittest.main()