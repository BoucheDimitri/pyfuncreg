from data import processing
from data import loading
from data import degradation
import os
from functional_regressors import kernels
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

X, Y = loading.load_raw_speech_dataset(os.getcwd() + "/data/dataspeech/raw/")
key = "VEL"

Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
        X, Y, shuffle_seed=543, n_train=300, normalize_domain=True, normalize_values=True)
Ytrain_ext, Ytrain, Ytest_ext, Ytest = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

# def func(x, gamma):
#     return np.tan(x) - 2 * gamma * x / (x ** 2 - gamma ** 2)
#
# def func_prime(x, gamma):
#     return 1 / (np.cos(x) ** 2) + 2 * gamma * (gamma ** 2 + x ** 2) / (x ** 2 - gamma ** 2) ** 2

def func(x, gamma):
    return (1 / np.tan(x)) - 0.5 * (x / gamma - gamma / x)

def func_prime(x, gamma):
    return - 1 / (np.sin(x) ** 2) - 0.5 * (1 / gamma + gamma / x ** 2)

def find_root(alpha=1, gamma=1):
    a, b = (2 * alpha - 1) * 0.5 * np.pi,  (2 * alpha + 1) * 0.5 * np.pi
    a0, b0 = a, alpha * np.pi
    root0 = root_scalar(func, args=(gamma, ), method="newton", fprime=func_prime, x0=0.5 * (a0 + b0),
                       bracket=(a0, b0), maxiter=1000)
    if a <= root0.root <= b:
        return root0
    else:
        a1, b1 = alpha * np.pi, b
        root1 = root_scalar(func, args=(gamma,), method="newton", fprime=func_prime, x0=0.5 * (a1 + b1),
                            bracket=(a1, b1), maxiter=1000)
        if a <= root1.root <= b:
            return root1
        else:
            raise Warning("Root outside of desired range")

root = find_root(alpha=10, gamma=100)

alpha = 0
gamma = 100
epsilon = 1e-10
x = np.linspace((2 * alpha - 1) * 0.5 * np.pi, (2 * alpha + 1) * 0.5 * np.pi, 1000)
y = [func(t, gamma) for t in x]

plt.plot(x, y)

gamma = 0.1

lapker = kernels.LaplaceScalarKernel(band=(1 / gamma), normalize=False)

kernel_sigmas = np.ones(13)
gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kernel_sigmas]
multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)

theta_grid = np.expand_dims(Ytrain_ext[0][0], axis=1)
Kout = lapker(theta_grid, theta_grid)
uout, vout = np.linalg.eigh(Kout)
uout = np.flip(uout)
vout = np.flip(vout, axis=1)

Kin = multi_ker(Xtrain, Xtrain)
uin, vin = np.linalg.eigh(Kin)
uin = np.flip(uin)
vin = np.flip(vin, axis=1)

kappa = 10
n = Kin.shape[0]
V = []

for i in range(n):
    for s in range(kappa):
        V.append(np.kron(np.expand_dims(vin[:, i], axis=1), np.expand_dims(vout[:, s], axis=0)))

V = np.array(V)
U = np.kron(uin, uout[:kappa].T).flatten()

Ymat = [Ytrain_ext[1][i] for i in range(n)]

scalar_prods = np.sum(np.multiply(V, Ymat), axis=(1, 2))

regu = 1
coefs = scalar_prods * (1 / (U + regu))
coefs = coefs[:, np.newaxis, np.newaxis]

C = (coefs * V).sum(axis=0)
















class SeparableOVKRidgeFunctional:
    """
    Discrete approximation of FKRR with separable kernel using Sylvester solver

    Parameters
    ----------
    regu : float
        The regularization parameter
    kernel_in : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns an array_like object with shape = [n_samples_x1, n_samples_x0].
    kernel_out : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns an array_like object with shape = [n_samples_x1, n_samples_x0].
    approx_locs : array_like
        The discretization space to use
    center_output : bool
        Should the outputs be centered upon training
    """
    def __init__(self, regu, kernel_in, kernel_out, approx_locs, center_output=False):
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.regu = regu
        self.approx_locs = np.squeeze(approx_locs)
        self.Kout = (1 / self.approx_locs.shape[0]) * self.kernel_out(np.expand_dims(self.approx_locs, axis=1),
                                                                      np.expand_dims(self.approx_locs, axis=1))
        self.smoother = smoothing.LinearInterpSmoother()
        self.alpha = None
        self.X = None
        self.Ymean = None
        self.center_output = center_output

    def fit(self, X, Y, Kin=None, return_cputime=False):
        # Memorize training input data
        self.X = X
        # Compute mean func from output data
        self.Ymean = disc_fd.mean_func(*Y)
        # Center discrete output data if relevant and put it in discrete general form
        if self.center_output:
            Ycentered = disc_fd.center_discrete(*Y, self.Ymean)
            Ycentered = disc_fd.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd.to_discrete_general(*Y)
        # Extract functions from centered output data using linear interpolation
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(*Ycentered)
        Yfunc = smoother_out.get_functions()
        # Evaluate those functions on the discretization grid
        Yeval = np.array([f(self.approx_locs) for f in Yfunc])
        # Compute input kernel matrix
        if Kin is None:
            Kin = self.kernel_in(X, X)
        # Compute representer coefficients
        n = len(X)
        m = self.Kout.shape[0]
        start = time.process_time()
        self.alpha = sb04qd(n, m, Kin / (self.regu * n), self.Kout, Yeval / (self.regu * n))
        end = time.process_time()
        if return_cputime:
            return end - start

    def predict(self, Xnew):
        Knew = self.kernel_in(self.X, Xnew)
        Ypred = (self.Kout.dot(self.alpha.T.dot(Knew.T))).T
        if self.center_output:
            Ymean_evals = self.Ymean(self.approx_locs)
            return Ypred.reshape((len(Xnew), len(self.approx_locs))) + Ymean_evals.reshape((1, len(self.approx_locs)))
        else:
            return Ypred.reshape((len(Xnew), len(self.approx_locs)))

    def predict_func(self, Xnew):
        Ypred = self.predict(Xnew)
        rep_locs = [np.expand_dims(self.approx_locs, axis=1) for i in range(len(Xnew))]
        self.smoother.fit(rep_locs, Ypred)
        return self.smoother.get_functions()

    def predict_evaluate(self, Xnew, locs):
        funcs = self.predict_func(Xnew)
        return np.array([func(locs) for func in funcs]).squeeze()

    def predict_evaluate_diff_locs(self, Xnew, Ylocs, return_cputime=False):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Ylocs[i])))
        return preds