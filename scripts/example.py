import sys
sys.path.append('../')
import recoengi
import recoengi.cf as cf

from scipy import sparse

# Simulate 100000 users with their preferences over 100 products. 
M = sparse.random(100000, 100, density=0.1, format='csr')

# Initialize PM 
pm = cf.PM(M)
# Compute the similarity matrix 
pm.computeSimilarityMatrix(bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = 10)

