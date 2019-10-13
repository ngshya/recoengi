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
dtf_train_orig = pickle.load(open("sampledata/movie_ratings_train.pickle", "rb"))
dtf_test = pickle.load(open("sampledata/movie_ratings_test.pickle", "rb"))


#%%
dtf_train = dtf_train_orig.set_index(["userId", "movieId"])["rating"].unstack(fill_value=0.0).rename_axis([None], axis=1).reset_index(drop=False)
dtf_train.index = dtf_train.userId
dtf_train = dtf_train.drop(["userId"], axis=1)
dtf_train.columns = ["film_"+str(x) for x in dtf_train.columns]


#%%
M = sparse.csc_matrix(dtf_train)
colnames = pd.Series(dtf_train.columns)
rownames = pd.Series(dtf_train.index)


#%%
array_dict_conf = [{
    "target": "film_"+str(x), 
    "target_type": "classification", 
    "threshold": 2.9,
    "features": np.setdiff1d(colnames, ["film_"+str(x)]), 
    "nfolds": 2,
    "n_estimators": 100,
    "max_depth": 10
} for x in range(4179)]
# 4179


#%%
output = cv.cvmMultiRun(array_dict_conf, M, colnames, rownames, npool=8)


#%%
dtf_pred = pd.DataFrame(np.array(output).transpose()).unstack().reset_index(drop=False)
dtf_pred.columns = ["movieId", "userId", "predicted_score"]


#%%
print("Training set shape: " + str(dtf_train_orig.shape))
print("Test set shape: " + str(dtf_test.shape))
dtf_train = pd.merge(dtf_train_orig, dtf_pred, on=["userId", "movieId"], how="inner")
dtf_test = pd.merge(dtf_test, dtf_pred, on=["userId", "movieId"], how="inner")
print("Training set shape: " + str(dtf_train.shape))
print("Test set shape: " + str(dtf_test.shape))


#%%
fpr, tpr, thresholds = metrics.roc_curve(y_true=(dtf_train.rating>2.9)+1, y_score=dtf_train.predicted_score, pos_label=2)
print("AUC on training set: " + str(metrics.auc(fpr, tpr)) + ".")
fpr, tpr, thresholds = metrics.roc_curve(y_true=(dtf_test.rating>2.9)+1, y_score=dtf_test.predicted_score, pos_label=2)
print("AUC on test set: " + str(metrics.auc(fpr, tpr)) + ".")

#%%
