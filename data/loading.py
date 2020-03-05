import pandas as pd
import numpy as np
import os
import pickle


def load_dti(path, shuffle_seed=0):
    # Set random seed
    np.random.seed(shuffle_seed)
    # Load the data
    cca = pd.read_csv(path + "cca_DTI_1st_visit.csv").iloc[:, 1:]
    rcst = pd.read_csv(path + "rcst_DTI_1st_visit.csv").iloc[:, 1:]
    n = cca.shape[0]
    # Shuffle index
    shuffle_inds = np.random.choice(n, n, replace=False)
    # Extract in numpy form and apply shuffle
    try:
        cca = cca.to_numpy()[shuffle_inds, 0:]
        rcst = rcst.to_numpy()[shuffle_inds, 0:]
    except AttributeError:
        cca = cca.values[shuffle_inds, 0:]
        rcst = rcst.values[shuffle_inds, 0:]
    return cca, rcst


def load_speech_dataset(path=os.getcwd() + "/data/dataspeech/processed/"):
    with open(path + "Xtrain.pkl", "rb") as inp:
        Xtrain = pickle.load(inp)
    with open(path + "Ytrain.pkl", "rb") as inp:
        Ytrain = pickle.load(inp)
    with open(path + "Xtest.pkl", "rb") as inp:
        Xtest = pickle.load(inp)
    with open(path + "Ytest.pkl", "rb") as inp:
        Ytest = pickle.load(inp)
    return Xtrain, Ytrain, Xtest, Ytest