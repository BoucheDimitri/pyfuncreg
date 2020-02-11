import numpy as np
import python_speech_features


def process_dti_dataset(cca, rcst, n_train=70, normalize01=True, interpout=True,
                        pad_train_input=True, pad_mode_input="symmetric",
                        pad_width_input=((0, 0), (93, 93)), pad_train_output=True,
                        pad_mode_output="symmetric", pad_width_output=((0, 0), (55, 55))):
    dimx, dimy = cca.shape[1], rcst.shape[1]
    n = cca.shape[0]
    if normalize01:
        norma_in, norma_out = dimx, dimy
    else:
        norma_in, norma_out = 1, 1
    cca_train, cca_test = cca[:n_train, :], cca[n_train:, :]
    rcst_train, rcst_test = rcst[:n_train, :], rcst[n_train:, :]
    full_input_locs = (1 / norma_in) * np.arange(dimx)
    full_output_locs = (1 / norma_out) * np.arange(dimy)
    if pad_train_input:
        cca_train = np.pad(cca_train, pad_width=pad_width_input, mode=pad_mode_input)
        full_input_locs_pad = (1 / norma_in) * np.arange(- pad_width_input[1][0], dimx + pad_width_input[1][1])
    else:
        full_input_locs_pad = full_input_locs
    if pad_train_output:
        rcst_train = np.pad(rcst_train, pad_width=pad_width_output, mode=pad_mode_output)
        full_output_locs_pad = (1 / norma_out) * np.arange(- pad_width_output[1][0], dimy + pad_width_output[1][1])
    else:
        full_output_locs_pad = full_output_locs
    Xtrain_locs, Xtrain_evals, Ytrain_locs, Ytrain_evals = [], [], [], []
    Xtest_locs, Xtest_evals, Ytest_locs, Ytest_evals = [], [], [], []
    for i in range(n_train):
        if interpout:
            ylocs = (1 / norma_out) * (np.argwhere(~ np.isnan(rcst_train[i])).squeeze() - pad_width_output[1][0])
            Ytrain_locs.append(full_output_locs_pad)
            Ytrain_evals.append(np.interp(full_output_locs_pad, ylocs, rcst_train[i][~ np.isnan(rcst_train[i])]))
        else:
            Ytrain_evals.append(rcst_train[i][~ np.isnan(rcst_train[i])])
            Ytrain_locs.append((1 / norma_out) * (np.argwhere(~ np.isnan(rcst_train[i])).squeeze() - pad_width_output[1][0]))
        xlocs = (1 / norma_in) * (np.argwhere(~ np.isnan(cca_train[i])).squeeze() - pad_width_input[1][0])
        Xtrain_locs.append(full_input_locs_pad)
        Xtrain_evals.append(np.interp(full_input_locs_pad, xlocs, cca_train[i][~ np.isnan(cca_train[i])]))
    for i in range(n - n_train):
        xlocs = (1 / norma_in) * np.argwhere(~ np.isnan(cca_test[i])).squeeze()
        Xtest_locs.append(full_input_locs)
        Xtest_evals.append(np.interp(full_input_locs, xlocs, cca_test[i][~ np.isnan(cca_test[i])]))
        Ytest_evals.append(rcst_test[i][~ np.isnan(rcst_test[i])])
        Ytest_locs.append((1 / norma_out) * np.argwhere(~ np.isnan(rcst_test[i])).squeeze())
    return (Xtrain_locs, Xtrain_evals), (Ytrain_locs, Ytrain_evals), (Xtest_locs, Xtest_evals), (Ytest_locs, Ytest_evals)


def process_speech_dataset(X, Y, duration="max", pad_mode="symmetric",
                           shuffle_seed=784, n_train=300, normalize01_output=True):
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
    if normalize01_output:
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
    return Xtrain, Ytrain, Xtest, Ytest
