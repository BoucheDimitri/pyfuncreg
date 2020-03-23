import numpy as np


class NoNanWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        evals = self.func(x)
        return np.nan_to_num(evals, nan=0)


def mean_function(func_list):
    """
    Compute the mean function of a list of functions

    Parameters
    ----------
    func_list: iterable of functions
        The functions

    Returns
    -------
    function
        The mean function
    """
    def mean_func(x):
        evals = np.array([func(x) for func in func_list]).squeeze()
        return evals.mean(axis=0)
    return mean_func


def diff_function(func1, func2):
    """
    Difference of two functions

    Parameters
    ----------
    func1: function
        The first function
    func2: function
        The second function
    Returns
    -------
    function
        The difference between the functions
    """
    def diff_func(x):
        return func1(x) - func2(x)
    return diff_func


def diff_function_list(func_list, func):
    """
    Substract the same function to a list of functions

    Parameters
    ----------
    func_list: iterable of functions
    func: function

    Returns
    -------
    iterable
        The list of function with func substracted
    """
    return [diff_function(f, func) for f in func_list]


def weighted_sum_function(coefs, func_list):
    """
    Weighted sum of functions

    Parameters
    ----------
    coefs: array-like
        The weights
    func_list: iterable of functions
        The functions

    Returns
    -------
    function
        Weighted sum of the functions
    """
    def weighted_sum_func(x):
        evals = np.array([coefs[i] * func_list[i](x) for i in range(len(coefs))]).squeeze()
        return np.sum(evals, axis=0)
    return weighted_sum_func


def product_function(func1, func2):
    """
    Compute the mean function of a list of functions

    Parameters
    ----------
    func_list: iterable of functions
        The functions

    Returns
    -------
    function
        The mean function
    """
    def prod_func(x):
        return func1(x) * func2(x)
    return prod_func