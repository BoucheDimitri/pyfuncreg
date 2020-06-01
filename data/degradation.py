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


def downsample_output_nan_ext(Y, Yext, remove_frac, random_state):
    n = len(Y[0])
    Ydown = Y[0], Y[1].copy()
    Yext_down = Yext[0], Yext[1].copy()
    for i in range(n):
        n_evals = len(Y[0][i])
        n_remove = int(remove_frac * n_evals)
        inds = random_state.choice(np.arange(n_evals), size=n_remove, replace=False)
        for j in inds:
            Ydown[1][i][j] = np.nan
        n_evals_ext = len(Yext[0][i])
        full_inds = np.pad(np.arange(n_evals), pad_width=(0, n_evals_ext), mode="symmetric")
        for j in range(n_evals_ext):
            if full_inds[j] in inds:
                Yext_down[1][i][j] = np.nan
    return Ydown, Yext_down