import pandas as pd
import numpy as np
from scipy.io import wavfile
import os


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


def load_speech_dataset_bis(path):
    row_dict = {"LP": 0, "LA": 1, "TBCL": 2, "TBCD": 3,  "VEL": 5, "GLO": 6, "TTCL": 7, "TTCD": 8}
    words = list(set([word.split(".")[0] for word in os.listdir(path)]))
    X = []
    Y = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
    for word in words:
        rate, signal = wavfile.read(path + word + ".wav")
        length = signal.shape[0] / rate
        ylocs = np.expand_dims(np.arange(0, length - 0.005, 0.005), axis=1)
        output_func_mat = pd.read_csv(path + word + ".csv", header=None)
        X.append(signal)
        try:
            output_func_mat = output_func_mat.to_numpy()
        except AttributeError:
            output_func_mat = output_func_mat.values
        for key in Y.keys():
            Y[key][1].append(output_func_mat[row_dict[key]])
            Y[key][0].append(ylocs)
    return X, Y