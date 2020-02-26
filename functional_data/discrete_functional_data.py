import numpy as np

from functional_data import smoothing
from functional_data import functional_algebra


def extrapolated_mean(Ylocs, Yevals):
    """
    Functions are firstly linearly extrapolated and then the obtained functions are averaged. Advised when
    the locations are shifting for each observations.

    Parameters
    ----------
    Ylocs: iterable of array-like
        The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
    Yevals: iterable of array-like
        The observations len = n_samples and for the i-th sample, Yevals[i] has shape = [n_observations_i, ]

    Returns
    -------
    function
        The mean function
    """
    linear_smoother = smoothing.LinearInterpSmoother()
    linear_smoother.fit(Ylocs, Yevals)
    return functional_algebra.mean_function(linear_smoother.get_functions())


def missing_sameloc_mean(Ylocs, Yevals):
    """
    Compute the mean directly on the sample ignoring nans and then extrapolate it to a function, advised for the case
    where most locations are in common with missing data.

    Parameters
    ----------
    Ylocs: iterable of array-like
        The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
    Yevals: iterable of array-like
        The observations len = n_samples and for the i-th sample, Yevals[i] has shape = [n_observations_i, ]

    Returns
    -------
    function
        The mean function
    """
    n = len(Ylocs)
    lens = np.array([len(Ylocs[i]) for i in range(n)])
    full_locs = Ylocs[np.argmax(lens)]
    dimy = np.max(lens)
    temp = np.full((n, dimy), np.nan)
    for i in range(n):
        n_missing = 0
        for j in range(dimy):
            if full_locs[j] in Ylocs[i]:
                temp[i, j] = Yevals[i][j - n_missing]
            else:
                n_missing += 1
    return smoothing.LinearInterpSmoother.interp_function(full_locs, np.nanmean(temp, axis=0))


MODES = {"full_extrapolate": extrapolated_mean, "samelocs_missing": missing_sameloc_mean}


def mean(Ylocs, Yevals, mode):
    return MODES[mode](Ylocs, Yevals)


def substract_function(Ylocs, Yevals, func):
    """
    Substract function to discrete functional sample

    Parameters
    ----------
    Ylocs: iterable of array-like
        The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
    Yevals: iterable of array-like
        The observations len = n_samples and for the i-th sample, Yevals[i] has shape = [n_observations_i, ]
    func: callable
        The function to substract

    Returns
    -------
    iterable of array-like
        The evalution at the input locations with the evaluations of func substracted

    """
    Yevals_sub = []
    for i in range(len(Yevals)):
        Yevals_sub.append(Yevals[i] - func(Ylocs[i]))
    return Yevals_sub




