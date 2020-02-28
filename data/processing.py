import numpy as np

# Sameloc possibly with missing data

# Shifting grid

# def normalize_domain01_samelocs_missing(Xlocs)


def normalize_domain(Xlocs, locmin, locmax):
    return [(Xlocs[i] - locmin) / locmax for i in range(len(Xlocs))]

def normalize_values():
    pass


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