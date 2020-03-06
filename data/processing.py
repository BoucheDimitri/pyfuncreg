import numpy as np
import python_speech_features

from data import loading


def process_dti(cca, rcst, n_train=70, normalize01=True, interp_input=True):
    # Add sampling locations
    locs_cca = np.arange(0, cca.shape[1])
    locs_rcst = np.arange(0, rcst.shape[1])
    # Use interpolation for filling NaNs in the input data if relevant
    if interp_input:
        for i in range(len(cca)):
            cca[i] = np.interp(locs_cca,
                               locs_cca[np.argwhere(~ np.isnan(cca[i])).squeeze()],
                               cca[i][np.argwhere(~ np.isnan(cca[i])).squeeze()])
    # Normalize locations to [0, 1] if relevant
    if normalize01:
        locs_cca = (1 / cca.shape[1]) * np.arange(0, cca.shape[1])
        locs_rcst = (1 / rcst.shape[1]) * np.arange(0, rcst.shape[1])
    return (locs_cca, cca[:n_train]), (locs_rcst, rcst[:n_train]), \
           (locs_cca, cca[n_train:]), (locs_rcst, rcst[n_train:])


RATE = 10000
OUTPUT_PACE = 0.005


def normalize_domain_max_duration(X):
    max_duration = np.max([x.shape[0] / RATE for x in X])
    norma = (1 / (max_duration - OUTPUT_PACE))
    return norma


def append_processed_data_point(X, Y, Xout, Yout, i, normalize_domain):
    # Signal length
    length = X[i].shape[0] / RATE
    # Locations of sampling of the output function
    norma = normalize_domain_max_duration(X) if normalize_domain else 1
    ylocs = norma * np.arange(0, length - OUTPUT_PACE, OUTPUT_PACE)
    # Compute MFCCs
    mfccs = python_speech_features.base.mfcc(X[i], samplerate=RATE, winlen=0.010, winstep=OUTPUT_PACE, numcep=13)
    Xout.append(mfccs)
    for vt in loading.VOCAL_TRACTS:
        Yout[vt][0].append(ylocs)
        Yout[vt][1].append(Y[vt][i])


def normalize_output_values(Ytrain, Ytest):
    Ytrain_norm = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytest_norm = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    norm_vals = dict()
    n_train = len(Ytrain[loading.VOCAL_TRACTS[0]][1])
    n_test = len(Ytest[loading.VOCAL_TRACTS[0]][1])
    for vt in loading.VOCAL_TRACTS:
        m = np.min(np.concatenate(Ytrain[vt][1]))
        M = np.max(np.concatenate(Ytrain[vt][1]))
        a = 2 / (M - m)
        b = 1 - a * M
        norm_vals[vt] = (a, b)
    for i in range(n_train):
        for vt in loading.VOCAL_TRACTS:
            Ytrain_norm[vt][1].append(norm_vals[vt][0] * Ytrain[vt][1][i] + norm_vals[vt][1])
            Ytrain_norm[vt][0].append(Ytrain[vt][0][i])
    for i in range(n_test):
        for vt in loading.VOCAL_TRACTS:
            Ytest_norm[vt][1].append(norm_vals[vt][0] * Ytest[vt][1][i] + norm_vals[vt][1])
            Ytest_norm[vt][0].append(Ytest[vt][0][i])
    return Ytrain_norm, Ytest_norm


def process_speech(X, Y, shuffle_seed=784, n_train=300, normalize_domain=True, normalize_values=True):
    # Initialize containers
    Xtrain, Xtest = list(), list()
    Ytrain = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    Ytest = {vt: [list(), list()] for vt in loading.VOCAL_TRACTS}
    # Shuffle index
    n_samples = len(X)
    inds = np.arange(n_samples)
    np.random.seed(shuffle_seed)
    np.random.shuffle(inds)
    # Fill data
    for i in inds[:n_train]:
        append_processed_data_point(X, Y, Xtrain, Ytrain, i, normalize_domain)
    for i in inds[n_train:]:
        append_processed_data_point(X, Y, Xtest, Ytest, i, normalize_domain)
    if normalize_values:
        Ytrain_norm, Ytest_norm = normalize_output_values(Ytrain, Ytest)
        return Xtrain, Ytrain_norm, Xtest, Ytest_norm
    else:
        return Xtrain, Ytrain, Xtest, Ytest