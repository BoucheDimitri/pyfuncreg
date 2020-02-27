import numpy as np


def mse(ytrue, ypred):
    scores = np.array([np.mean(np.linalg.norm(np.squeeze(ytrue[i]) - np.squeeze(ypred[i]))**2) for i in range(len(ytrue))])
    return np.mean(scores)

