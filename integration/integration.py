from functional_data import functional_algebra

from scipy import integrate
import numpy as np


# def func_scalar_prod(func1, func2, domain):
#     prod = functional_algebra.product_function(func1, func2)
#     return integrate.quad(prod, domain[0, 0], domain[0, 1])

def func_scalar_prod(func1, func2, domain, nevals=1000):
    locs = np.linspace(domain[0, 0], domain[0, 1], nevals)
    evals1, evals2 = func1(locs), func2(locs)
    evals1, evals2 = np.nan_to_num(evals1, nan=0), np.nan_to_num(evals2, nan=0)
    return np.mean(evals1 * evals2)