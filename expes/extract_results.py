import pickle
import os
import numpy as np


def mean_variance_result_dti(path):
    scores_test = []
    for f in os.listdir(path):
        with open(path + "/" + f, "rb") as inp:
            _1, _2, score_test = pickle.load(inp)
        scores_test.append(score_test)
    return np.mean(scores_test), np.std(scores_test)

