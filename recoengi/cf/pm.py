from scipy import sparse
from sklearn.preprocessing import normalize
from .cosimtop import *

class PM:
    '''
    Preferences Matrix Class
    '''
    
    def __init__(self, M):
        '''
        Initialization. 
        :param M: Preferences matrix, each row represents a user, each column represents the product.  
        '''
        # Computations are performed in sparse matrices in order to hold more data. 
        self.M = sparse.csr_matrix(M)

    def computeSimilarityMatrix(self, bln_bin = False, bln_norm = True, flt_ths = 0.0, ntop = None):
        '''
        Compute the cosine similarity matrix.
        :param bln_bin: Binarize each matrix entry? If True, each entry will be binarized according to the flt_ths. 
        :param bln_norm: Should the output matrix entries be constrained in the interval [-1, +1]? 
        :param flt_ths: Threshold used by the binarization. 
        :param ntop: How many most similar users to keep for each user? If None, every similarity coefficient will be kept. Set this parameter to take advantage of the sparse matrix memory optimization. 
        '''
        if ntop is None:
            ntop = self.M.shape[0]
        self.S = self.M.copy()
        if bln_bin:
            self.S.data = (self.S.data > flt_ths) + 0.0
        if bln_norm:
            self.S = normalize(self.S, norm='l2', axis=1)
        self.S = cosimtop(self.S, self.S.transpose(), ntop = ntop)