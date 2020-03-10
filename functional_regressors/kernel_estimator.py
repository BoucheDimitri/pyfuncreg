import numpy as np

from functional_data import smoothing
from functional_data import functional_algebra
from functional_data import discrete_functional_data as disc_fd


def gaussian_window(x):
    return (1 / (np.sqrt(2 * np.pi))) * np.exp(- x ** 2 / 2)


class KernelEstimatorFunc:
    """
    Parameters
    ----------
    kernel: callable
        The window function
    bandwidth: float
        The window width parameter
    """
    def __init__(self, kernel, bandwidth):
        self.bandwidth = bandwidth
        self.X = None
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.Yfunc = None

    def fit(self, X, Y):
        self.X = X
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(Y[0], Y[1])
        self.Yfunc = smoother_out.get_functions()

    def predict(self, Xnew):
        n = self.X.shape[0]
        m = Xnew.shape[0]
        gram_mat = np.zeros((m, n))
        for j in range(m):
            gram_mat[j, :] = np.linalg.norm(self.X - Xnew[j], axis=1) ** 2
        Knew = np.apply_along_axis(self.kernel, 0, gram_mat / self.bandwidth)
        W = Knew / np.expand_dims(np.sum(Knew, axis=1), axis=1)
        return [functional_algebra.weighted_sum_function(W[i], self.Yfunc) for i in range(m)]

    def predict_evaluate(self, Xnew, locs):
        pred_funcs = self.predict(Xnew)
        return np.array([func(locs) for func in pred_funcs])


class KernelEstimatorStructIn:
    """
    Version of the kernel estimator with kernelized input
    Parameters
    ----------
    kernel: functional_regressors.kernels.ScalarKernel
        The kernel for the input data
    center_output: bool
        Should the output data be centered upon training
    """
    def __init__(self, kernel, center_output=False):
        self.kernel = kernel
        self.X = None
        self.Yfunc = None
        self.Ymean = None
        self.center_output = center_output

    def fit(self, X, Y):
        self.X = X
        self.Ymean = disc_fd.mean_func(*Y)
        if self.center_output:
            Ycentered = disc_fd.center_discrete(*Y, self.Ymean)
            Ycentered = disc_fd.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd.to_discrete_general(*Y)
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(*Ycentered)
        self.Yfunc = smoother_out.get_functions()

    def predict(self, Xnew):
        m = len(Xnew)
        Knew = self.kernel(self.X, Xnew)
        W = Knew / np.expand_dims(np.sum(Knew, axis=1), axis=1)
        # return W
        return [functional_algebra.weighted_sum_function(W[i], self.Yfunc) for i in range(m)]

    def predict_evaluate(self, Xnew, locs):
        pred_funcs = self.predict(Xnew)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(locs.squeeze(), self.Ymean[0], self.Ymean[1]), axis=0)
            return np.array([func(locs) for func in pred_funcs]) + extrapolate_mean
        else:
            return np.array([func(locs) for func in pred_funcs])

    def predict_evaluate_diff_locs(self, Xnew, Ylocs):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(self.predict_evaluate([Xnew[i]], Ylocs[i]).squeeze())
        return preds


