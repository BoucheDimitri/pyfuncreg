import numpy as np


def mean_function(func_list):
    def mean_func(x):
        evals = np.array([func(x) for func in func_list]).squeeze()
        return evals.mean(axis=0)
    return mean_func


def diff_function(func1, func2):
    def diff_func(x):
        return func1(x) - func2(x)
    return diff_func


def diff_function_list(func_list, func):
    return [diff_function(f, func) for f in func_list]


def weighted_sum_function(coefs, func_list):
    def weighted_sum_func(x):
        evals = np.array([coefs[i] * func_list[i](x) for i in range(len(coefs))]).squeeze()
        return np.sum(evals, axis=0)
    return weighted_sum_func