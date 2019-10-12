import sys
sys.path.append('../')
sys.path.append('.')
import recoengi
import recoengi.cf as cf

from scipy import sparse
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")


# Simulate 100000 users with their preferences over 100 products. 
M = sparse.random(30000, 32, density=0.3, format='csr')

# Initialize PM 
cfm = cf.CFM(M)

# Steps by steps
# Compute the similarity matrix 
# cfm.computeSimilarityMatrix(bln_bin=False, bln_norm=True, flt_ths=0.0, ntop=10, flt_lb=-1)
# Compute the scores
# cfm.computeScores()
# Compute the amounts
# cfm.computeAmounts()
# Compute the performances
# cfm.computePerformances()

# ... OR
cfm.computeEverything(bln_bin=False, bln_norm=True, flt_ths=0.0, ntop=10, flt_lb=-1)

