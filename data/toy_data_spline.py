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

# Correlated version params
F_MAX = 11
C_MAX = 1
ALPHA = 0.8
LAMB = 0.5
# ######################################################################################################################


def cos_atom(a):
    def funca(x):
        return np.cos(a * x)
    return funca


def sin_atom(a):
    def funca(x):
        return np.sin(a * x)
    return funca


def correlated_freqs_draws(f_max, size, alpha=0.8, seed_state=784):
    n_samples = size[0]
    n_freqs = size[1]
    odd_freqs = [2 * i - 1 for i in range(1, f_max // 2 + 1)]
    even_freqs = [2 * i for i in range(1, f_max // 2 + 1)]
    n_freqs_odds = n_freqs // 2
    draws_odd = []
    random_state = np.random.RandomState(seed_state)
    for i in range(n_samples):
        draws_odd.append(random_state.choice(odd_freqs, replace=False, size=n_freqs_odds))
    draws_even = [[] for i in range(n_samples)]
    for i in range(n_samples):
        remaining = set(even_freqs)
        for j in range(n_freqs_odds):
            d = random_state.binomial(1, alpha)
            if d == 1:
                f = draws_odd[i][j] + 1
                draws_even[i].append(f)
                remaining = remaining.difference({f})
            else:
                f = random_state.choice(list(remaining))
                draws_even[i].append(f)
                remaining = remaining.difference({f})
    draws_odd, draws_even = np.array(draws_odd), np.array(draws_even)
    return np.sort(np.concatenate((draws_odd, draws_even), axis=1), axis=1)


def correlated_coefs_draws(sorted_freqs, c_max=1, lamb=0.4, seed_state=784):
    coefs = np.zeros(sorted_freqs.shape)
    random_state = np.random.RandomState(seed_state)
    for i in range(sorted_freqs.shape[0]):
        coefs[i, 0] = random_state.uniform(-c_max, c_max)
        for j in range(1, sorted_freqs.shape[1]):
            if sorted_freqs[i, j] % 2 == 0 and sorted_freqs[i, j] - sorted_freqs[i, j-1] == 1:
                u = random_state.uniform(-c_max, c_max)
                coefs[i, j] = (1 - lamb) * coefs[i, j-1] + lamb * u
            else:
                coefs[i, j] = random_state.uniform(-c_max, c_max)
    return coefs


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


def generate_toy_spline_correlated(n_samples, dom_input, dom_output, n_locs_input, n_locs_output,
                                   n_freqs, f_max, c_max, alpha=0.8, lamb=0.5, width=2, seed=0):
    locs_input = np.linspace(dom_input[0, 0], dom_input[0, 1], n_locs_input)
    locs_output = np.linspace(dom_output[0, 0], dom_output[0, 1], n_locs_output)
    freqs = correlated_freqs_draws(f_max, size=(n_samples, n_freqs), alpha=alpha, seed_state=seed)
    coefs = correlated_coefs_draws(freqs, c_max=c_max, lamb=lamb, seed_state=seed)
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


def get_toy_data_correlated(n_train):
    X, Y = generate_toy_spline_correlated(N_SAMPLES + N_TEST, DOM_INPUT, DOM_OUTPUT, N_LOCS_INPUT, N_LOCS_OUTPUT,
                                          N_FREQS, F_MAX, C_MAX, ALPHA, LAMB, WIDTH, SEED_TOY)
    Xtrain = np.array([X[1][n] for n in range(n_train)])
    Ytrain = ([np.expand_dims(Y[0][n], axis=1) for n in range(n_train)], [Y[1][n] for n in range(n_train)])
    Xtest = np.array([X[1][n] for n in range(N_SAMPLES, N_SAMPLES + N_TEST)])
    Ytest = ([np.expand_dims(Y[0][n], axis=1) for n in range(N_SAMPLES, N_SAMPLES + N_TEST)],
             [Y[1][n] for n in range(N_SAMPLES, N_SAMPLES + N_TEST)])
    return Xtrain, Ytrain, Xtest, Ytest