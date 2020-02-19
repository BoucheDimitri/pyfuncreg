import pandas as pd
import numpy as np
import os
import pickle


def load_dti(path, shuffle_seed=0):
    np.random.seed(shuffle_seed)
    cca = pd.read_csv(path + "cca_DTI_1st_visit.csv").iloc[:, 1:]
    rcst = pd.read_csv(path + "rcst_DTI_1st_visit.csv").iloc[:, 1:]
    n = cca.shape[0]
    shuffle_inds = np.random.choice(n, n, replace=False)
    try:
        cca = cca.to_numpy()[shuffle_inds, 0:]
        rcst = rcst.to_numpy()[shuffle_inds, 0:]
    except AttributeError:
        cca = cca.values[shuffle_inds, 0:]
        rcst = rcst.values[shuffle_inds, 0:]
    return cca, rcst


def load_processed_speech_dataset(path=os.getcwd() + "/data/dataspeech/processed/",
                                  pad_width=None, pad_mode="symmetric", normalize_output=True):
    with open(path + "Xtrain.pkl", "rb") as inp:
        Xtrain = pickle.load(inp)
    with open(path + "Ytrain.pkl", "rb") as inp:
        Ytrain = pickle.load(inp)
    with open(path + "Xtest.pkl", "rb") as inp:
        Xtest = pickle.load(inp)
    with open(path + "Ytest.pkl", "rb") as inp:
        Ytest = pickle.load(inp)
    if pad_width is not None:
        Ytrain_padded = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
                         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
        for key in Ytrain.keys():
            Ytrain_sub_array = np.array(Ytrain[key][1]).squeeze()
            leny = Ytrain_sub_array.shape[1]
            Ytrain_sub_array = np.pad(Ytrain_sub_array,
                                      pad_width=((0, 0), (pad_width[0] * leny, pad_width[1] * leny)),
                                      mode=pad_mode)
            n = Ytrain_sub_array.shape[0]
            Ylocs_padded = Ytrain[key][0][0]
            locs = Ytrain[key][0][0]
            for i in range(pad_width[0]):
                Ylocs_padded = np.concatenate((-i - 1 + locs, Ylocs_padded))
            for i in range(pad_width[1]):
                Ylocs_padded = np.concatenate((Ylocs_padded, locs + i + 1))
            for j in range(n):
                Ytrain_padded[key][0].append(Ylocs_padded)
                Ytrain_padded[key][1].append(Ytrain_sub_array[j])
    else:
        Ytrain_padded = Ytrain
    if normalize_output:
        Ytrain_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
                         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
        Ytest_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
                         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
        for key in Ytrain.keys():
            m = np.min(np.array(Ytrain_padded[key][1]))
            M = np.max(np.array(Ytrain_padded[key][1]))
            a = 2 / (M - m)
            b = 1 - a * M
            for j in range(len(Ytrain[key][0])):
                Ytrain_normalized[key][1].append(a * Ytrain_padded[key][1][j] + b)
                Ytrain_normalized[key][0].append(Ytrain[key][0][j])
            for j in range(len(Ytest[key][1])):
                Ytest_normalized[key][1].append(a * Ytest[key][1][j] + b)
                Ytest_normalized[key][0].append(Ytest[key][0][j])
    else:
        Ytrain_normalized = Ytrain_padded
        Ytest_normalized = Ytest
    return Xtrain, Ytrain_normalized, Xtest, Ytest_normalized