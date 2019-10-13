#%%
import recoengi
import recoengi.cf as cf

import pandas as pd
from scipy import sparse
from sklearn import metrics
import pickle
import pkg_resources
import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(message)s")


#%%
dtf_train = pickle.load(open(pkg_resources.resource_filename('recoengi', 'sampledata/movie_ratings_train.pickle'), "rb"))
dtf_test = pickle.load(open(pkg_resources.resource_filename('recoengi', 'sampledata/movie_ratings_test.pickle'), "rb"))


#%%
M = sparse.csr_matrix((dtf_train.rating, (dtf_train.userId, dtf_train.movieId)))


#%%
cfm = cf.CFM(M)
cfm.computeEverything(bln_bin=False, bln_norm=True, flt_ths=2.9, ntop=64, flt_lb=-1)


#%%
dtf_pred = pd.DataFrame(pd.Series(dict(cfm.SCORES.todok().items()))).reset_index(drop=False)
dtf_pred.columns = ["userId", "movieId", "predicted_score"]
dtf_train = pd.merge(dtf_train, dtf_pred, on=["userId", "movieId"], how="inner")
dtf_test = pd.merge(dtf_test, dtf_pred, on=["userId", "movieId"], how="inner")


#%%
fpr, tpr, thresholds = metrics.roc_curve(y_true=(dtf_train.rating>2.9)+1, y_score=dtf_train.predicted_score, pos_label=2)
print("AUC on training set: " + str(metrics.auc(fpr, tpr)) + ".")
fpr, tpr, thresholds = metrics.roc_curve(y_true=(dtf_test.rating>2.9)+1, y_score=dtf_test.predicted_score, pos_label=2)
print("AUC on test set: " + str(metrics.auc(fpr, tpr)) + ".")