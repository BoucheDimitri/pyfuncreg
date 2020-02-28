import numpy as np

from functional_data import smoothing
from functional_data import functional_algebra


class DiscreteRegular1D:

    def __init__(self, ylocs, Yobs):
        """

        Parameters
        ----------
        ylocs: array_like, shape = [n_locations_full, ]
        Yobs: array_like, shape = [n_samples, n_locations_full]
        """
        self.ylocs_full = ylocs
        self.pace = ylocs[1] - ylocs[0]
        self.Yobs_full = Yobs
        self.n_samples = len(Yobs)
        self.n_obs = len(ylocs)

    def to_discrete_general(self):
        Ylocs, Yobs = list(), list()
        for i in range(len(self.Yobs_full)):
            Ylocs.append(self.ylocs_full[np.argwhere(~ np.isnan(self.Yobs_full[i])).squeeze()])
        return Ylocs, [self.Yobs_full[i] for i in range(len(self.Yobs_full))]

    def get_extended_version(self, mode="symmetric", repeats=(0, 0)):
        padded_locs = [self.ylocs_full + i * len(self.ylocs_full) for i in range(repeats[0] + repeats[1] + 1)]
        Yobs_full_extended = np.pad(self.Yobs_full, mode=mode,
                                    pad_width=((0, 0), (repeats[0] * self.n_obs, repeats[1] * self.n_obs)))
        return DiscreteRegular1D(
            np.concatenate(padded_locs) - repeats[0] * (self.ylocs_full[-1] - self.ylocs_full[0] + self.pace),
            Yobs_full_extended)

    def mean_discrete(self):
        return self.ylocs_full, np.nanmean(self.Yobs_full, axis=0)

    def get_mean_func(self):
        ylocs_full, yobs_mean = self.mean_discrete()
        return smoothing.LinearInterpSmoother.interp_function(ylocs_full, yobs_mean)







def extrapolated_mean(Ylocs, Yobs):
    """
    Functions are firstly linearly extrapolated and then the obtained functions are averaged. Advised when
    the locations are shifting for each observations.

    Parameters
    ----------
    Ylocs: iterable of array-like
        The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
    Yobs: iterable of array-like
        The observations len = n_samples and for the i-th sample, Yobs[i] has shape = [n_observations_i, ]

    Returns
    -------
    function
        The mean function
    """
    linear_smoother = smoothing.LinearInterpSmoother()
    linear_smoother.fit(Ylocs, Yobs)
    return functional_algebra.mean_function(linear_smoother.get_functions())


def missing_sameloc_mean(Ylocs, Yobs):
    """
    Compute the mean directly on the sample ignoring nans and then extrapolate it to a function, advised for the case
    where most locations are in common with missing data.

    Parameters
    ----------
    Ylocs: iterable of array-like
        The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
    Yobs: iterable of array-like
        The observations len = n_samples and for the i-th sample, Yobs[i] has shape = [n_observations_i, ]

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
                temp[i, j] = Yobs[i][j - n_missing]
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




