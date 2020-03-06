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
    return mean_func_samelocs(Ylocs, Yobs) if samelocs else  mean_func_difflocs(Ylocs, Yobs)
