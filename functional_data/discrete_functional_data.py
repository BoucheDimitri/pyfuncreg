from functional_data import smoothing
from functional_data import functional_algebra

import numpy as np


def test_samelocs(Ylocs):
    for i in range(len(Ylocs) - 1):
        if len(Ylocs[i]) != len(Ylocs[i]):
            return False
    for i in range(len(Ylocs) - 1):
        if not np.all(np.equal(Ylocs[i], Ylocs[i + 1])):
            return False
    return True


def discrete_to_func_lininterp(Ylocs, Yobs):
    smoother = smoothing.LinearInterpSmoother()
    smoother.fit(Ylocs, Yobs)
    Yfunc = smoother.get_functions()
    return Yfunc


def mean_func_samelocs(Ylocs, Yobs):
    mean_discrete = np.nanmean(np.concatenate([np.expand_dims(y, axis=0) for y in Yobs]), axis=0)
    return smoothing.LinearInterpSmoother.interp_function(Ylocs[0], mean_discrete)


def mean_func_difflocs(Ylocs, Yobs):
    # mean_func = functional_algebra.mean_function(discrete_to_func_lininterp(Ylocs, Yobs))
    mean_func = functional_algebra.MeanFunction(discrete_to_func_lininterp(Ylocs, Yobs))
    return mean_func


def mean_func(Ylocs, Yobs):
    samelocs = test_samelocs(Ylocs)
    return mean_func_samelocs(Ylocs, Yobs) if samelocs else mean_func_difflocs(Ylocs, Yobs)


def center_discrete(Ylocs, Yobs, mean_func):
    Yobs_centered = list()
    for i in range(len(Ylocs)):
        Yobs_centered.append(Yobs[i] - mean_func(Ylocs[i]))
    return Ylocs, Yobs_centered


def extend_signal_samelocs(ylocs, Yobs, mode, repeats, add_locs=True):
    n_obs = len(ylocs)
    pace = ylocs[1] - ylocs[0]
    ylocs_extended = [ylocs + i * (ylocs[-1] - ylocs[0] + pace) for i in range(repeats[0] + repeats[1] + 1)]
    Yobs_extended = np.pad(Yobs, mode=mode, pad_width=((0, 0), (repeats[0] * n_obs, repeats[1] * n_obs)))
    if add_locs:
        return set_locs(np.concatenate(ylocs_extended) - repeats[0] * (ylocs[-1] - ylocs[0] + pace), Yobs_extended)
    else:
        return np.concatenate(ylocs_extended) - repeats[0] * (ylocs[-1] - ylocs[0] + pace), Yobs_extended


def set_locs(ylocs, Yobs):
    return [ylocs for i in range(len(Yobs))], Yobs


def extend_locs_if_samelocs(Ylocs, Yobs):
    if np.squeeze(Ylocs[0]).shape == ():
        return set_locs(Ylocs, Yobs)
    else:
        return Ylocs, Yobs


def to_discrete_general(Ylocs, Yobs):
    # Remove NaNs
    Ylocs_dg, Yobs_dg = list(), list()
    n_samples = len(Yobs)
    for i in range(n_samples):
        Ylocs_dg.append(Ylocs[i][np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
        Yobs_dg.append(Yobs[i][np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
    return Ylocs_dg, Yobs_dg


def to_function_linearinterp(Ylocs, Yobs):
    smoother = smoothing.LinearInterpSmoother()
    smoother.fit(Ylocs, Yobs)
    return smoother.get_functions()


