import sys
sys.path.append('../')
import recoengi
import recoengi.cf as cf

from scipy import sparse

# Simulate 100000 users with their preferences over 100 products. 
M = sparse.random(4, 5, density=0.1, format='csr')

# Initialize PM 
pm = cf.CFM(M)
# Compute the similarity matrix 
pm.computeSimilarityMatrix(bln_bin=False, bln_norm=True, flt_ths=0.0, ntop=10, flt_lb=-1)
# Compute the scores
pm.computeScores()