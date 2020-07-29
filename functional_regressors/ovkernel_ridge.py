# IMPORTANT: slycot should be installed using the following command:
# $conda install -c conda-forge slycot
from slycot import sb04qd
import numpy as np
import time
import numba

from ..functional_data import smoothing, discrete_functional_data as disc_fd


class SeparableOVKRidge:
    """
    Ovk ridge with separable kernel using Sylvester solver

    Parameters
    ----------
    regu : float
        The regularization parameter
    kernel : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns an array_like object with shape = [n_samples_x1, n_samples_x0].
    B : array_like, shape = [n_output_features, n_output_features]
        The matrix encoding the similarity between the outputs
    """
    def __init__(self, regu, kernel, B):
        self.kernel = kernel
        self.B = B
        self.regu = regu
        self.K = None
        self.alpha = None
        self.X = None

    def fit(self, X, Y, K=None):
        self.X = X
        if K is not None:
            self.K = K
        else:
            self.K = self.kernel(X, X)
        n = len(X)
        m = len(self.B)
        self.alpha = sb04qd(n, m, self.K / (self.regu * n), self.B, Y / (self.regu * n))

    def predict(self, Xnew):
        Knew = self.kernel(self.X, Xnew)
        preds = (self.B.dot(self.alpha.T.dot(Knew.T))).T
        return preds


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


# @numba.njit
# def invert_kroneig(alpha, Vin, uin, Vout, uout, Yeval, regu, n, t, neig_in, neig_out):
#     vin = np.zeros((n, 1), dtype=numba.float64)
#     vout = np.zeros((1, t), dtype=numba.float64)
#     for i in range(neig_in):
#         for s in range(neig_out):
#             vin[:, 0] = Vin[:, i]
#             vout[0, :] = Vout[:, s]
#             V = np.kron(vin, vout)
#             alpha += (1 / (uin[i] * uout[s] + regu)) * (np.multiply(V, Yeval)).sum() * V

def invert_kroneig(alpha, Vin, uin, Vout, uout, Yeval, regu, n, t, neig_in, neig_out):
    vin = np.zeros((n, 1), dtype=np.float64)
    vout = np.zeros((1, t), dtype=np.float64)
    for i in range(neig_in):
        for s in range(neig_out):
            vin[:, 0] = Vin[:, i]
            vout[0, :] = Vout[:, s]
            V = np.kron(vin, vout)
            alpha += (1 / (uin[i] * uout[s] + regu)) * (np.multiply(V, Yeval)).sum() * V


class SeparableOVKRidgeFunctionalEigsolve:
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
    def __init__(self, regu, kernel_in, kernel_out, neig_in, neig_out, approx_locs, center_output=False):
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.neig_out = neig_out
        self.neig_in = neig_in
        self.regu = regu
        self.approx_locs = np.squeeze(approx_locs)
        self.Kout = (1 / self.approx_locs.shape[0]) * self.kernel_out(np.expand_dims(self.approx_locs, axis=1),
                                                                      np.expand_dims(self.approx_locs, axis=1))
        uout, vout = np.linalg.eigh(self.Kout)
        self.uout, self.vout = np.flip(uout), np.flip(vout, axis=1)
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
        start = time.process_time()
        # Eigen decomposition of kernel matrix
        uin, vin = np.linalg.eigh(Kin)
        uin, vin = np.flip(uin), np.flip(vin, axis=1)
        n = Kin.shape[0]
        m = self.Kout.shape[0]
        # Less memory usage
        self.alpha = np.zeros((n, m), dtype="float64")
        invert_kroneig(self.alpha, vin, uin, self.vout, self.uout, Yeval, self.regu, n, m, self.neig_in, self.neig_out)
        # self.alpha = np.zeros((n, m))
        # for i in range(n):
        #     for s in range(self.kappa):
        #         V = np.kron(np.expand_dims(vin[:, i], axis=1), np.expand_dims(self.vout[:, s], axis=0))
        #         self.alpha += (1 / (uin[i] * self.uout[s] + self.regu)) * (np.multiply(V, Yeval)).sum() * V
        # With numpy
        # Complete Eigenvectors
        # V = []
        # for i in range(n):
        #     for s in range(self.kappa):
        #         V.append(np.kron(np.expand_dims(vin[:, i], axis=1), np.expand_dims(self.vout[:, s], axis=0)))
        # V = np.array(V)
        # # Complete Eigenvalues
        # U = np.kron(uin, self.uout[:self.kappa].T).flatten()
        # scalar_prods = np.sum(np.multiply(V, Yeval), axis=(1, 2))
        # coefs = scalar_prods * (1 / (U + self.regu))
        # coefs = coefs[:, np.newaxis, np.newaxis]
        # # Representer coefficients
        # self.alpha = (coefs * V).sum(axis=0)
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


def invert_kroneigapprox(alpha, Vin, uin, Vout, uout, Yeval, regu, n, t, neig):
    vin = np.zeros((n, 1), dtype=np.float64)
    vout = np.zeros((1, t), dtype=np.float64)
    eigvals = np.kron(uin, uout)
    inds_sort = np.flip(np.argsort(eigvals))
    for j in range(neig):
        # Infer back indexes
        i = inds_sort[j] // t
        s = inds_sort[j] % t
        vin[:, 0] = Vin[:, i]
        vout[0, :] = Vout[:, s]
        V = np.kron(vin, vout)
        alpha += (1 / (uin[i] * uout[s] + regu)) * (np.multiply(V, Yeval)).sum() * V


class SeparableOVKRidgeFunctionalEigapprox:
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
    def __init__(self, regu, kernel_in, kernel_out, neig, approx_locs, center_output=False):
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.neig = neig
        self.regu = regu
        self.approx_locs = np.squeeze(approx_locs)
        self.Kout = (1 / self.approx_locs.shape[0]) * self.kernel_out(np.expand_dims(self.approx_locs, axis=1),
                                                                      np.expand_dims(self.approx_locs, axis=1))
        uout, vout = np.linalg.eigh(self.Kout)
        self.uout, self.vout = np.flip(uout), np.flip(vout, axis=1)
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
        start = time.process_time()
        # Eigen decomposition of kernel matrix
        uin, vin = np.linalg.eigh(Kin)
        uin, vin = np.flip(uin), np.flip(vin, axis=1)
        n = Kin.shape[0]
        m = self.Kout.shape[0]
        # Less memory usage
        self.alpha = np.zeros((n, m), dtype="float64")
        invert_kroneigapprox(self.alpha, vin, uin, self.vout, self.uout, Yeval, self.regu, n, m, self.neig)
        # self.alpha = np.zeros((n, m))
        # for i in range(n):
        #     for s in range(self.kappa):
        #         V = np.kron(np.expand_dims(vin[:, i], axis=1), np.expand_dims(self.vout[:, s], axis=0))
        #         self.alpha += (1 / (uin[i] * self.uout[s] + self.regu)) * (np.multiply(V, Yeval)).sum() * V
        # With numpy
        # Complete Eigenvectors
        # V = []
        # for i in range(n):
        #     for s in range(self.kappa):
        #         V.append(np.kron(np.expand_dims(vin[:, i], axis=1), np.expand_dims(self.vout[:, s], axis=0)))
        # V = np.array(V)
        # # Complete Eigenvalues
        # U = np.kron(uin, self.uout[:self.kappa].T).flatten()
        # scalar_prods = np.sum(np.multiply(V, Yeval), axis=(1, 2))
        # coefs = scalar_prods * (1 / (U + self.regu))
        # coefs = coefs[:, np.newaxis, np.newaxis]
        # # Representer coefficients
        # self.alpha = (coefs * V).sum(axis=0)
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