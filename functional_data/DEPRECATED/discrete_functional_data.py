import numpy as np

from functional_data import smoothing
from functional_data import functional_algebra

# # Modes correspondances:
# # "mode1" is short for "discrete_samelocs_regular_1d"
#
#
# def extend_mode1(ylocs, Yobs, mode, repeats):
#     n_obs = len(ylocs)
#     pace = ylocs[1] - ylocs[0]
#     ylocs_extended = [ylocs + i * (ylocs[-1] - ylocs[0] + pace) for i in range(repeats[0] + repeats[1] + 1)]
#     Yobs_extended = np.pad(Yobs, mode=mode, pad_width=((0, 0), (repeats[0] * n_obs, repeats[1] * n_obs)))
#     return np.concatenate(ylocs_extended) - repeats[0] * (ylocs[-1] - ylocs[0] + pace), Yobs_extended
#
#
# def to_discrete_general_mode1(ylocs, Yobs):
#     Ylocs_dg, Yobs_dg = list(), list()
#     n_samples = len(Yobs)
#     for i in range(n_samples):
#         Ylocs_dg.append(ylocs[np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
#         Yobs.append(Yobs[i][np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
#     return Ylocs_dg, Yobs_dg
#


class DiscreteSamelocsRegular1D:

    def __init__(self, ylocs, Yobs):
        """
        Discretely sampled functional data with 1D inputs on a regular grid possibly with missing data

        Parameters
        ----------
        ylocs : array_like, shape = [n_locations_full, ]
        Yobs : array_like, shape = [n_samples, n_locations_full]
        """
        self.ylocs, self.Yobs = ylocs, Yobs
        self.pace = ylocs[1] - ylocs[0]
        self.n_samples = len(Yobs)
        self.n_obs = len(ylocs)

    @staticmethod
    def extend_signal(ylocs, Yobs, mode, repeats):
        n_obs = len(ylocs)
        pace = ylocs[1] - ylocs[0]
        ylocs_extended = [ylocs + i * (ylocs[-1] - ylocs[0] + pace) for i in range(repeats[0] + repeats[1] + 1)]
        Yobs_extended = np.pad(Yobs, mode=mode, pad_width=((0, 0), (repeats[0] * n_obs, repeats[1] * n_obs)))
        return np.concatenate(ylocs_extended) - repeats[0] * (ylocs[-1] - ylocs[0] + pace), Yobs_extended

    @staticmethod
    def to_discrete_general(ylocs, Yobs):
        """
        Put data in general discretized function form

        Returns
        -------
        tuple
            (Ylocs, Yobs) both with len = n_samples and for  1 <= i <= n_samples,
            Ylocs[i] and Yobs[i] are array-like both of shape = [n_observations_i, ]
        """
        Ylocs_dg, Yobs_dg = list(), list()
        n_samples = len(Yobs)
        for i in range(n_samples):
            Ylocs_dg.append(ylocs[np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
            Yobs_dg.append(Yobs[i][np.argwhere(~ np.isnan(Yobs[i])).squeeze()])
        return Ylocs_dg, Yobs_dg

    def discrete_general(self):
        """
        Put data in general discretized function form

        Returns
        -------
        tuple
            (Ylocs, Yobs) both with len = n_samples and for  1 <= i <= n_samples,
            Ylocs[i] and Yobs[i] are array-like both of shape = [n_observations_i, ]
        """
        return DiscreteSamelocsRegular1D.to_discrete_general(self.ylocs, self.Yobs)

    def extended_version(self, mode="symmetric", repeats=(0, 0)):
        """
        Extends signal and return a corresponding new class instance

        Parameters
        ----------
        mode : {"symmetric"}
            Extension mode
        repeats : tuple of int, len = 2
            Number of time to repeat the whole signal before (first tuple component) and after (second tuple component)

        Returns
        -------
        DiscreteRegular1D
            A new class instance with extended signal
        """
        ylocs_extended, Yobs_extended = DiscreteSamelocsRegular1D.extend_signal(self.ylocs, self.Yobs, mode, repeats)
        return DiscreteSamelocsRegular1D(ylocs_extended, Yobs_extended)

    def mean_discrete(self):
        """
        Compute means of discretized functions ignoring NaNs

        Returns
        -------
        array_like
            The mean, has shape = [n_observations, ]
        """
        return self.ylocs, np.nanmean(self.Yobs, axis=0)

    def mean_func(self):
        """
        Return mean function using linear interpolation (and extrapolation)

        Returns
        -------
        function
            The mean function
        """
        ylocs_full, yobs_mean = self.mean_discrete()
        return smoothing.LinearInterpSmoother.interp_function(ylocs_full, yobs_mean)

    def centered_discrete_general(self):
        Ylocs, Yobs = self.discrete_general()
        Ymean = self.mean_func()
        Yobs_centered = list()
        for i in range(self.n_samples):
            Yobs_centered.append(Yobs[i] - Ymean(Ylocs[i]))
        return Ylocs, Yobs_centered


class DiscreteGeneral:

    def __init__(self, Ylocs, Yobs):
        self.Ylocs = Ylocs
        self.Yobs = Yobs
        self.n_samples = len(Yobs)
        self.n_obs = [len(Yobs[i]) for i in range(self.n_samples)]

    @staticmethod
    def to_func_linearinterp(Ylocs, Yobs):
        smoother = smoothing.LinearInterpSmoother()
        smoother.fit(Ylocs, Yobs)
        Yfunc = smoother.get_functions()
        return Yfunc

    def func_linearinterp(self):
        return DiscreteGeneral.to_func_linearinterp(self.Ylocs, self.Yobs)

    def mean_func(self):
        """
        Return mean function using linear interpolation (and extrapolation)

        Returns
        -------
        function
            The mean function
        """
        return functional_algebra.mean_function(self.func_linearinterp())

    def centered_func_linearinterp(self):
        return functional_algebra.diff_function_list(self.func_linearinterp(), self.mean_func())

    @staticmethod
    def to_discrete_general(Ylocs, Yobs):
        return Ylocs, Yobs

    def discrete_general(self):
        return self.Ylocs, self.Yobs


DATA_TYPES = {'discrete_samelocs_regular_1d': DiscreteSamelocsRegular1D,
              'discrete_general': DiscreteGeneral}


def wrap_functional_data(Y, key):
    return DATA_TYPES[key](Y[0], Y[1])


def to_discrete_general(Y, data_format):
    return DATA_TYPES[data_format].to_discrete_general(Y[0], Y[1])


def preprocess_data(data, signal_ext, center=False, data_format="discrete_samelocs_regular_1d"):
    data_wrapped = wrap_functional_data(data, data_format)
    # Memorize mean function before signal extension
    if center:
        data_mean = data_wrapped.mean_func()
    else:
        data_mean = None
    # Extends the signal if relevant
    if signal_ext is not None:
        data_wrapped_extended = data_wrapped.extended_version(signal_ext[0], signal_ext[1])
    else:
        data_wrapped_extended = data_wrapped
    # Center with extended signal if relevant
    if center:
        return data_mean, data_wrapped_extended.centered_discrete_general()
    else:
        return data_mean, data_wrapped_extended.discrete_general()




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



