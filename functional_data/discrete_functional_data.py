from functional_data import smoothing
from functional_data import functional_algebra

import numpy as np


def test_samelocs(Ylocs):
    samelocs = True
    for i in range(len(Ylocs) - 1):
        if not np.all(np.equal(Ylocs[i], Ylocs[i + 1])):
            samelocs = False
    return samelocs


def discrete_to_func_lininterp(Ylocs, Yobs):
    smoother = smoothing.LinearInterpSmoother()
    smoother.fit(Ylocs, Yobs)
    Yfunc = smoother.get_functions()
    return Yfunc


def mean_func_samelocs(Ylocs, Yobs):
    mean_discrete = np.nanmean(np.concatenate([np.expand_dims(y, axis=0) for y in Yobs]), axis=0)
    return smoothing.LinearInterpSmoother.interp_function(Ylocs[0], mean_discrete)


def mean_func_difflocs(Ylocs, Yobs):
    mean_func = functional_algebra.mean_function(discrete_to_func_lininterp(Ylocs, Yobs))
    return mean_func


def mean_func(Ylocs, Yobs):
    samelocs = test_samelocs(Ylocs)
    return mean_func_samelocs(Ylocs, Yobs) if samelocs else mean_func_difflocs(Ylocs, Yobs)


def center_discrete(Ylocs, Yobs, mean_func):
    Yobs_centered = list()
    for i in range(len(Ylocs)):
        Yobs_centered.append(Yobs[i] - mean_func(Ylocs[i]))
    return Ylocs, Yobs_centered


def extend_signal_samelocs(ylocs, Yobs, mode, repeats):
    n_obs = len(ylocs)
    pace = ylocs[1] - ylocs[0]
    ylocs_extended = [ylocs + i * (ylocs[-1] - ylocs[0] + pace) for i in range(repeats[0] + repeats[1] + 1)]
    Yobs_extended = np.pad(Yobs, mode=mode, pad_width=((0, 0), (repeats[0] * n_obs, repeats[1] * n_obs)))
    return np.concatenate(ylocs_extended) - repeats[0] * (ylocs[-1] - ylocs[0] + pace), Yobs_extended


def set_locs(ylocs, Yobs):
    return [ylocs for i in range(len(Yobs))], Yobs


