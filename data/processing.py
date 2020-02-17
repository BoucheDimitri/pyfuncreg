import numpy as np
import python_speech_features
import pickle
import os


def process_dti_dataset(cca, rcst, n_train=70, normalize01=True,
                        pad_mode_input="symmetric", pad_width_input=((0, 0), (0, 0)),
                        pad_mode_output="symmetric", pad_width_output=((0, 0), (0, 0))):
    # Evaluation grids dimensions
    dimx, dimy = cca.shape[1], rcst.shape[1]
    # Number of sample points
    n = cca.shape[0]
    # Normalization constants
    if normalize01:
        norma_in, norma_out = dimx, dimy
    else:
        norma_in, norma_out = 1, 1
    # Divide between train and test
    cca_train, cca_test = cca[:n_train, :], cca[n_train:, :]
    rcst_train, rcst_test = rcst[:n_train, :], rcst[n_train:, :]
    # Pad training inputs
    cca_train = np.pad(cca_train, pad_width=pad_width_input, mode=pad_mode_input)
    cca_test = np.pad(cca_test, pad_width=pad_width_input, mode=pad_mode_input)
    full_input_locs_pad = (1 / norma_in) * np.arange(-pad_width_input[1][0], dimx + pad_width_input[1][1])
    # Pad training outputs
    rcst_train = np.pad(rcst_train, pad_width=pad_width_output, mode=pad_mode_output)
    # Initialize containers
    Xtrain_locs, Xtrain_evals, Ytrain_locs, Ytrain_evals = [], [], [], []
    Xtest_locs, Xtest_evals, Ytest_locs, Ytest_evals = [], [], [], []
    for i in range(n_train):
        # We interpolate linearily the inputs
        # Locations for the inputs that are not NaNs
        xlocs = (1 / norma_in) * (np.argwhere(~ np.isnan(cca_train[i])).squeeze() - pad_width_input[1][0])
        # Interpolate linearily to get observations at all locations
        Xtrain_evals.append(np.interp(full_input_locs_pad, xlocs, cca_train[i][~ np.isnan(cca_train[i])]))
        # Add all locations to current data point
        Xtrain_locs.append(full_input_locs_pad)
        #
        Ytrain_evals.append(rcst_train[i][~ np.isnan(rcst_train[i])])
        Ytrain_locs.append((1 / norma_out) * (np.argwhere(~ np.isnan(rcst_train[i])).squeeze() - pad_width_output[1][0]))
    for i in range(n - n_train):
        xlocs = (1 / norma_in) * (np.argwhere(~ np.isnan(cca_test[i])).squeeze() - pad_width_input[1][0])
        Xtest_locs.append(full_input_locs_pad)
        Xtest_evals.append(np.interp(full_input_locs_pad, xlocs, cca_test[i][~ np.isnan(cca_test[i])]))
        Ytest_evals.append(rcst_test[i][~ np.isnan(rcst_test[i])])
        Ytest_locs.append((1 / norma_out) * np.argwhere(~ np.isnan(rcst_test[i])).squeeze())
    return (Xtrain_locs, Xtrain_evals), (Ytrain_locs, Ytrain_evals), (Xtest_locs, Xtest_evals), (Ytest_locs, Ytest_evals)


def process_speech_dataset(X, Y, duration="max", pad_mode="symmetric",
                           shuffle_seed=784, n_train=300, normalize01_domain=True, normalize_range=True):
    n = len(X)
    durations = [len(x) / 10000 for x in X]
    if duration == "max":
        duration = np.max(durations)
    Xtrain = []
    Ytrain = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
    Xtest = []
    Ytest = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
    inds = np.arange(n)
    np.random.seed(shuffle_seed)
    np.random.shuffle(inds)
    ylocs = np.arange(0, duration - 0.005, 0.005)
    if normalize01_domain:
        norma = ylocs[-1]
    else:
        norma = 1
    ylocs *= 1 / norma
    leny = len(ylocs.squeeze())
    for i in range(n_train):
        lenx = len(X[inds[i]])
        duration_x = lenx / 10000
        n_signal_pad = int((duration / duration_x))
        xpad = np.pad(X[inds[i]], pad_width=(0, n_signal_pad * lenx), mode=pad_mode)
        xpad = xpad[:int(duration * 10000)]
        mfccs = python_speech_features.base.mfcc(xpad, samplerate=10000, winlen=0.010, winstep=0.005, numcep=13)
        Xtrain.append(mfccs)
        for key in Ytrain.keys():
            y = Y[key][1][inds[i]]
            ypad = np.pad(y, pad_width=(0, n_signal_pad * len(y)), mode=pad_mode)
            Ytrain[key][0].append(ylocs)
            Ytrain[key][1].append(ypad[:leny])
    for i in range(n_train, n):
        lenx = len(X[inds[i]])
        duration_x = lenx / 10000
        n_signal_pad = int((duration / duration_x))
        xpad = np.pad(X[inds[i]], pad_width=(0, n_signal_pad * lenx), mode=pad_mode)
        xpad = xpad[:int(duration * 10000)]
        mfccs = python_speech_features.base.mfcc(xpad, samplerate=10000, winlen=0.010, winstep=0.005, numcep=13)
        Xtest.append(mfccs)
        for key in Ytest.keys():
            Ytest[key][0].append((1 / norma) * Y[key][0][inds[i]])
            Ytest[key][1].append(Y[key][1][inds[i]])
    if normalize_range:
        Ytrain_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
                         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
        Ytest_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
                         "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
        for key in Ytrain.keys():
            m = np.min(np.array(Ytrain[key][1]))
            M = np.max(np.array(Ytrain[key][1]))
            a = 2 / (M - m)
            b = 1 - a * M
            for j in range(len(Ytrain[key][0])):
                Ytrain_normalized[key][1].append(a * Ytrain[key][1][j] + b)
                Ytrain_normalized[key][0].append(Ytrain[key][0][j])
            for j in range(len(Ytest[key][1])):
                Ytest_normalized[key][1].append(a * Ytest[key][1][j] + b)
                Ytest_normalized[key][0].append(Ytest[key][0][j])
    else:
        Ytrain_normalized = Ytrain
        Ytest_normalized = Ytest
    return Xtrain, Ytrain_normalized, Xtest, Ytest_normalized


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