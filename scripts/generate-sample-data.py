import pandas as pd
import numpy as np
import pickle


dtf_ratings = pd.read_csv("sampledata/ml-latest-small/ratings.csv", usecols=["userId", "movieId", "rating"])
dtf_tmp = dtf_ratings.groupby(["movieId"]).agg({"rating": "count"}).reset_index(drop=False)
dtf_tmp = dtf_tmp.loc[dtf_tmp.rating > 3,:]
dtf_ratings = dtf_ratings.loc[dtf_ratings.movieId.isin(dtf_tmp.movieId),:]

dtf_tmp = pd.DataFrame({"movieId": dtf_ratings.movieId.unique()})
dtf_tmp = dtf_tmp.reset_index(drop = False)
dtf_tmp = dtf_tmp.rename({"index": "movieId_new"}, axis = 1)
dtf_ratings = pd.merge(dtf_ratings, dtf_tmp, on = ["movieId"], how = "left").drop(["movieId"], axis=1).rename({"movieId_new": "movieId"}, axis=1)
dtf_ratings.userId = dtf_ratings.userId-1
# dtf_ratings.rating = (dtf_ratings.rating >= 3) + 0.0

tmp_bln_split = np.random.choice([True, False], size=dtf_ratings.shape[0], replace=True, p=[0.8, 0.2])
dtf_train = dtf_ratings.loc[tmp_bln_split, ["userId", "movieId", "rating"]]
dtf_test = dtf_ratings.loc[~tmp_bln_split, ["userId", "movieId", "rating"]]
dtf_test = dtf_test.loc[dtf_test.userId.isin(dtf_train.userId) & dtf_test.movieId.isin(dtf_train.movieId), :]

pickle.dump(dtf_train, open("sampledata/movie_ratings_train.pickle", "wb"))
pickle.dump(dtf_test, open("sampledata/movie_ratings_test.pickle", "wb"))