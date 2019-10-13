from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
import numpy as np
from scipy import sparse
import logging

def cvMrun(M, colnames, rownames, dict_conf, n_estimators=100, max_depth=10):
    
    '''

    '''

    M = sparse.csc_matrix(M)

    y = np.array(M[:, colnames.index[colnames == dict_conf["target"]]].todense()).flatten()
    if dict_conf["target_type"] == "classification":
        y = (y > dict_conf["threshold"]) + 0
    
    X = M[:, colnames.index[colnames.isin(dict_conf["features"])]]
    
    random_folds = np.random.choice(range(dict_conf["nfolds"]), size=len(y)) + 1
    
    predictions = np.zeros(len(y))

    if dict_conf["target_type"] == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=1102
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=1102
        ) 

    for k in range(dict_conf["nfolds"]):
        k = k + 1
        logging.debug("Target " + dict_conf["target"] + " | Fold " + str(k) + ".")
        bln_tmp = (random_folds != k)
        X_tmp = X[rownames.index[bln_tmp], ]
        y_tmp = y[bln_tmp]
        model.fit(X_tmp, y_tmp)
        if dict_conf["target_type"] == "classification":
            predictions_tmp = model.predict_proba(X[rownames.index[~bln_tmp], :])
            predictions[~bln_tmp] = predictions_tmp[:, model.classes_ == 1].flatten()
        else:
            predictions_tmp = model.predict(X[rownames.index[~bln_tmp], :])
            predictions[~bln_tmp] = predictions_tmp

    if dict_conf["target_type"] == "classification":
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y+1, y_score=predictions, pos_label=2)
        logging.debug("AUC on training set: " + str(metrics.auc(fpr, tpr)) + ".")
    else: 
        logging.debug("RMSE on training set: " + str( np.mean( (predictions-y)**2 )**(0.5) ) + ".")



class cvm:

    '''
    Cross-validated recommender system. 
    '''

    def __init__(self, dtf_data, dict_features): 
        self.data = dtf_data
        self.dict_features = dict_features