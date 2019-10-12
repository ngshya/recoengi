from scipy import sparse
from sklearn.preprocessing import normalize
from .cosimtop import *

class PM:
    
    def __init__(self, M):
        self.M = sparse.csr_matrix(M)

    def computeSimilarityMatrix(self, bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = 3):
        self.S = self.M.copy()
        if bln_bin:
            self.S.data = (self.S.data > flt_ths) + 0.0
        if bln_norm:
            self.S = normalize(self.S, norm='l2', axis=1)
        self.S = cosimtop(self.S, self.S.transpose(), ntop = ntop)