#%%
import sys
sys.path.append('../')
sys.path.append('.')
import recoengi
import recoengi.cv as cv

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import metrics
import pickle
import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(message)s")


#%%
dtf_train = pickle.load(open("sampledata/movie_ratings_train.pickle", "rb"))
dtf_test = pickle.load(open("sampledata/movie_ratings_test.pickle", "rb"))


#%%
dtf_train = dtf_train.set_index(["userId", "movieId"])["rating"].unstack(fill_value=0.0).rename_axis([None], axis=1).reset_index(drop=False)
dtf_train.index = dtf_train.userId
dtf_train = dtf_train.drop(["userId"], axis=1)
dtf_train.columns = ["film_"+str(x) for x in dtf_train.columns]


#%%
M = sparse.csc_matrix(dtf_train)
colnames = pd.Series(dtf_train.columns)
rownames = pd.Series(dtf_train.index)


#%%
dict_conf = {
    "target": "film_0", 
    "target_type": "regression", 
    "threshold": 2.9,
    "features": np.setdiff1d(colnames, ["film_0"]), 
    "nfolds": 3
}


#%%
cv.cvMrun(M, colnames, rownames, dict_conf)

#%%
