# IMPORTANT: slycot and then control should installed using the following commands:
# $conda install -c conda-forge slycot
# $conda install -c conda-forge control
from control import dlyap
from slycot import sb04qd
import numpy as np

from functional_data import smoothing
from functional_data import functional_algebra


class SeparableOVKRidge:
    """
    Ovk ridge with separable kernel using Sylvester solver

    Parameters
    ----------
    input_kernel: functional_regressors.kernels.ScalarKernel
        The input scalar kernel
    output_mat: array-like, shape = [n_output_features, n_output_features]
        The matrix encoding the similarity between the outputs
    lamb: float
        The regularization parameter
    """
    def __init__(self, input_kernel, output_mat, lamb):
        self.input_kernel = input_kernel
        self.B = output_mat
        self.lamb = lamb
        self.K = None
        self.alpha = None
        self.X = None

    def fit(self, X, Y, K=None):
        self.X = X
        if K is not None:
            self.K = K
        else:
            self.K = self.input_kernel(X, X)
        n = len(X)
        m = len(self.B)
        # self.alpha = sb04qd(n, m, -self.K/(self.lamb * n), self.B.T, Y/(self.lamb * n))
        self.alpha = sb04qd(n, m, self.K / (self.lamb * n), self.B, Y / (self.lamb * n))
        print(self.alpha.shape)
        # self.alpha = np.array(dlyap(-self.K/(self.lamb * n), self.B.T, Y/(self.lamb * n)))

    def predict(self, Xnew):
        Knew = self.input_kernel(self.X, Xnew)
        preds = (self.B.dot(self.alpha.T.dot(Knew.T))).T
        return preds

# TODO: FINIR ADAPTATION DE CA
class SeparableOVKRidgeFunctional:
    """
    Discrete approximation of FKRR Ovk ridge with separable kernel using Sylvester solver

    Parameters
    ----------
    regu: float
        The regularization parameter
    input_kernel: functional_regressors.kernels.ScalarKernel
        The input scalar kernel
    output_kernel: functional_regressors.kernels.ScalarKernel
        The output scalar kernel
    approx_locs: array-like
        The discretization space to use
    center_output : bool
        Should the outputs be centered upon training
    """
    def __init__(self, regu, input_kernel, output_kernel, approx_locs, center_output=False):
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.regu = regu
        self.approx_locs = np.squeeze(approx_locs)
        self.Kout = (1 / self.approx_locs.shape[0]) * self.output_kernel(np.expand_dims(self.approx_locs, axis=1),
                                                                         np.expand_dims(self.approx_locs, axis=1))
        self.smoother = smoothing.LinearInterpSmoother()
        self.alpha = None
        self.X = None
        self.Ymean = None
        self.center_output = center_output

    def fit(self, X, Y):
        self.X = X
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(Y[0], Y[1])
        Yfunc = smoother_out.get_functions()
        if self.center_output:
            self.Ymean = functional_algebra.mean_function(Yfunc)
            Yfunc = functional_algebra.diff_function_list(Yfunc, self.Ymean)
        Yeval = np.array([f(self.approx_locs) for f in Yfunc])
        Kin = self.input_kernel(X, X)
        n = len(X)
        m = self.Kout.shape[0]
        # self.alpha = np.array(dlyap(-Kin/(self.regu * n), self.Kout.T, Yeval/(self.regu * n)))
        self.alpha = sb04qd(n, m, Kin / (self.regu * n), self.Kout, Y / (self.regu * n))

    def predict(self, Xnew):
        Knew = self.input_kernel(self.X, Xnew)
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

    def predict_evaluate_diff_locs(self, Xnew, Ylocs):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Ylocs[i])))
        return preds
