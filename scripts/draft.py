import sys
sys.path.append('../')
import recoengi
import recoengi.cf as cf

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

A = np.array([[1.0,2], [1, 0], [2, 3]])

X = cf.PM(A)
X.M.todense()

X.computeSimilarityMatrix(bln_bin = True, bln_norm = True)
X.S.todense()

X.computeSimilarityMatrix(bln_bin = True, bln_norm = False)
X.S.todense()

X.computeSimilarityMatrix(bln_bin = False, bln_norm = True)
X.S.todense()

X.computeSimilarityMatrix(bln_bin = False, bln_norm = False)
X.S.todense()