import numpy as np
from scipy.interpolate import BSpline
import functools

from functional_data import functional_algebra
from functional_data import basis


# ########################### GLOBAL SETTING FOR TOY DATASET ###########################################################
N_TEST = 300
N_SAMPLES = 5000
N_FREQS = 4
DOM_INPUT = np.expand_dims(np.array([0, 2 * np.pi]), axis=0)
N_LOCS_INPUT = 200
N_LOCS_OUTPUT = 200
BOUNDS_COEFS = [-1, 1]
COEFS_DRAW_FUNC = functools.partial(np.random.uniform, BOUNDS_COEFS[0], BOUNDS_COEFS[1])
BOUNDS_FREQS = [1, 10]
FREQS_DRAW_FUNC = functools.partial(np.random.randint, BOUNDS_FREQS[0], BOUNDS_FREQS[1])
WIDTH = 2
SEED_TOY = 784
DOM_OUTPUT = np.expand_dims(np.array([BOUNDS_FREQS[0] - WIDTH/2, BOUNDS_FREQS[1] + WIDTH/2]), axis=0)
COEFS_ = functools.partial(np.random.normal, 0, 1)
# ######################################################################################################################


def cos_atom(a):
    def funca(x):
        return np.cos(a * x)
    return funca


def sin_atom(a):
    def funca(x):
        return np.sin(a * x)
    return funca


def centered_cubic_spline(a, width):
    pace = width / 4
    knots = np.array([a-2 * pace, a - pace, a, a + pace, a + 2 * pace])

    def spl(x):
        y = BSpline.basis_element(knots, extrapolate=False)(x)
        y[np.isnan(y)] = 0
        return y
    return spl


def generate_toy_spline(n_samples, dom_input, dom_output, n_locs_input, n_locs_output,
                        n_freqs, freqs_draw_func, coefs_draw_func, width=2, seed=0):
    locs_input = np.linspace(dom_input[0, 0], dom_input[0, 1], n_locs_input)
    locs_output = np.linspace(dom_output[0, 0], dom_output[0, 1], n_locs_output)
    np.random.seed(seed)
    freqs = freqs_draw_func((n_samples, n_freqs))
    np.random.seed(seed)
    coefs = coefs_draw_func((n_samples, n_freqs))
    basis_in = []
    basis_out = []
    for n in range(n_samples):
        basis_in.append(basis.BasisFromSmoothFunctions([cos_atom(a) for a in freqs[n]], 1, dom_input))
        basis_out.append(basis.BasisFromSmoothFunctions([centered_cubic_spline(freqs[n, i], width)
                                                         for i in range(n_freqs)], 1, dom_output))
    smooth_in = [functional_algebra.weighted_sum_function(coefs[n], basis_in[n]) for n in range(n_samples)]
    smooth_out = [functional_algebra.weighted_sum_function(coefs[n], basis_out[n]) for n in range(n_samples)]
    X = ([locs_input.copy() for i in range(n_samples)], [func(locs_input) for func in smooth_in])
    Y = ([locs_output.copy() for i in range(n_samples)], [func(locs_output) for func in smooth_out])
    return X, Y


def get_toy_data(n_train):
    X, Y = generate_toy_spline(N_SAMPLES + N_TEST, DOM_INPUT, DOM_OUTPUT, N_LOCS_INPUT, N_LOCS_OUTPUT, N_FREQS,
                               FREQS_DRAW_FUNC, COEFS_DRAW_FUNC, WIDTH, SEED_TOY)
    Xtrain = np.array([X[1][n] for n in range(n_train)])
    Ytrain = ([np.expand_dims(Y[0][n], axis=1) for n in range(n_train)], [Y[1][n] for n in range(n_train)])
    Xtest = np.array([X[1][n] for n in range(N_SAMPLES, N_SAMPLES + N_TEST)])
    Ytest = ([np.expand_dims(Y[0][n], axis=1) for n in range(N_SAMPLES, N_SAMPLES + N_TEST)],
             [Y[1][n] for n in range(N_SAMPLES, N_SAMPLES + N_TEST)])
    return Xtrain, Ytrain, Xtest, Ytest