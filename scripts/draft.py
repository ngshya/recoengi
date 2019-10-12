import sys
sys.path.append('../')
import recoengi
import recoengi.cf as cf

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

A = sparse.random(100000, 100, density=0.1, format='csr')

pm = cf.PM(A)
pm.computeSimilarityMatrix(bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = 10)
