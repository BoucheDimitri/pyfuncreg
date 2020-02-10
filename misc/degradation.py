import numpy as np


def add_noise_inputs(X, std, seed):
    np.random.seed(seed)
    return X + np.random.normal(0, std, X.shape)


def add_noise_outputs(Y, std, seed):
    n = len(Y[0])
    Ynoisy_out = []
    for i in range(n):
        np.random.seed(seed)
        Ynoisy_out.append(Y[1][i] + np.random.normal(0, std, Y[1][i].shape))
    return Y[0], Ynoisy_out


def downsample_output(Y, remove_frac, seed):
    n = len(Y[0])
    Ydown = [[], []]
    for i in range(n):
        np.random.seed(seed)
        n_evals = len(Y[0][i])
        n_remove = int(remove_frac * n_evals)
        inds = np.random.choice(np.arange(n_evals), size=n_evals - n_remove, replace=False)
        inds.sort()
        Ydown[0].append(Y[0][i][inds])
        Ydown[1].append(np.array([Y[1][i][j] for j in inds]))
    return Ydown
