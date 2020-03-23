from functional_data import functional_algebra

from scipy import integrate


def func_scalar_prod(func1, func2, domain):
    prod = functional_algebra.product_function(func1, func2)
    return integrate.quad(prod, domain[0, 0], domain[0, 1])