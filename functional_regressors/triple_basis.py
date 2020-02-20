import numpy as np
from sklearn import kernel_ridge

from functional_data import fpca
from functional_data import basis
from functional_data import functional_algebra
from functional_data import sparsely_observed
from functional_data import smoothing


class KernelRidge:
    """Wrapper for scikit learn kernel ridge with custom kernel"""
    def __init__(self, kernel, regu):
        self.kernel = kernel
        self.regu = regu
        self.reg = kernel_ridge.KernelRidge(alpha=self.regu, kernel="precomputed")
        self.X = None

    def fit(self, X, Y, K=None):
        self.X = X
        if K is None:
            K = self.kernel(X, X)
        self.reg.fit(K, Y)

    def predict(self, Xnew):
        Knew = self.kernel(self.X, Xnew)
        return self.reg.predict(Knew)


class TripleBasisEstimator:
    """
    Triple basis estimator

    Parameters
    ----------
    basis_in: functional_data.basis.Basis
        The input orthonormal basis
    basis_rffs: functional_data.basis.RandomFourierFeatures
        Random Fourier Features used for the approximation
    basis_out: functional_data.basis.Basis
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    non_padded_index: array-like, optional
        The full set of locations of observations, only needed if center_output is True
    """
    def __init__(self, basis_in, basis_rffs, basis_out, regu, non_padded_index=None, center_output=True):
        self.basis_in = basis_in
        self.basis_rffs = basis_rffs
        self.basis_out = basis_out
        self.regu = regu
        self.regressors = None
        self.center_output = center_output
        self.non_padded_index = non_padded_index
        self.full_output_locs = None
        self.Ymean = None

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        scalar_prods = np.array([eval_mats[i].T.dot((1/X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

    def fit(self, X, Y):
        if self.center_output:
            full_output_locs, Ymean = sparsely_observed.mean_missing(Y[0], Y[1])
            self.Ymean = Ymean[self.non_padded_index[0]:self.non_padded_index[1]]
            self.full_output_locs = full_output_locs[self.non_padded_index[0]:self.non_padded_index[1]]
            Ycentered = sparsely_observed.substract_missing(full_output_locs, Ymean, Y[0], Y[1])
        else:
            Ycentered = Y
        coefsX = TripleBasisEstimator.projection_coefs(X, self.basis_in)
        coefsY = TripleBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = smoothing.ExpandedRidge(self.regu, self.basis_rffs)
            reg.fit(coefsX, coefsY[:, prob])
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew):
        coefsXnew = TripleBasisEstimator.projection_coefs(Xnew, self.basis_in)
        preds = np.array([reg(coefsXnew) for reg in self.regressors]).T
        return preds

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(yin_new.squeeze(), self.full_output_locs, self.Ymean), axis=0)
            return pred_coefs.dot(basis_evals.T) + extrapolate_mean
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew[0])
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate(
                (np.expand_dims(Xnew[0][i], axis=0), np.expand_dims(Xnew[1][i], axis=0)),
                Yins_new[i])))
        return preds


class BiBasisEstimator:
    """
    Triple basis estimator with structured input

    Parameters
    ----------
    kernel: functional_regressors.kernels.ScalarKernel
        The input kernel
    basis_out: functional_data.basis.Basis
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    """
    def __init__(self, kernel, basis_out, regu, center_output=False):
        self.basis_out = basis_out
        self.kernel = kernel
        self.regu = regu
        self.regressors = None
        self.X = None
        self.Ymean = None
        self.center_output = center_output

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        scalar_prods = np.array([eval_mats[i].T.dot((1 / X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

    def fit(self, X, Y, K=None):
        if self.center_output:
            self.Ymean = Y[0][0], np.mean(np.array(Y[1]).squeeze(), axis=0)
            Ycentered = Y[0], [y - self.Ymean[1] for y in Y[1]]
        else:
            Ycentered = Y
        self.X = X
        coefsY = BiBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        if K is None:
            K = self.kernel(X, X)
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = KernelRidge(self.kernel, self.regu)
            reg.fit(X, coefsY[:, prob])
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew):
        preds = np.array([reg.predict(Xnew) for reg in self.regressors]).T
        return preds

    def predict_from_coefs(self, pred_coefs, yin_new):
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(yin_new.squeeze(), self.Ymean[0], self.Ymean[1]), axis=0)
            return pred_coefs.dot(basis_evals.T) + extrapolate_mean
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        pred_coefs = self.predict(Xnew)
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        return preds


class BiBasisEstimatorFpca:
    """
    Triple basis estimator with structured input and FPCA output basis

    Parameters
    ----------
    kernel: functional_regressors.kernels.ScalarKernel
        The input kernel
    regu: float
        Regularization parameter
    nfpca: int
        The number of function principal components to include
    nevals_fpca: int
        The number of evaluations to use for approximation of the FPCA
    center_output: bool, optional
        Should outputs be centered ?
    """
    def __init__(self, kernel, regu, nfpca, nevals_fpca=500, center_output=True):
        self.kernel = kernel
        self.regu = regu
        self.nfpca = nfpca
        self.fpca = None
        self.Ymean_func = None
        self.regressors = None
        self.X = None
        self.basis_out = None
        self.Ymean = None
        self.center_output = center_output
        self.nevals_fpca = nevals_fpca

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        # shape = (n_samples, n_basis)
        scalar_prods = np.array([eval_mats[i].T.dot((1 / X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

    @staticmethod
    def get_func_outputs(Y):
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(Y[0], Y[1])
        return smoother_out.get_functions()

    def intialize_dict(self, Yfunc, domain):
        self.fpca = fpca.FunctionalPCA(domain, self.nevals_fpca, smoothing.LinearInterpSmoother())
        self.fpca.fit(Yfunc)
        if self.center_output:
            self.basis_out = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.nfpca), 1, domain)
        else:
            self.basis_out = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.nfpca), 1, domain,
                                                            add_constant=True)

    def fit(self, X, Y, K=None):
        n = len(X)
        Yfunc = BiBasisEstimatorFpca.get_func_outputs(Y)
        self.Ymean_func = functional_algebra.mean_function(Yfunc)
        if self.center_output:
            Yfunc_centered = functional_algebra.diff_function_list(Yfunc, self.Ymean_func)
        else:
            Yfunc_centered = Yfunc
        domain = np.array([[Y[0][0][0], Y[0][0][-1]]])
        self.intialize_dict(Yfunc_centered, domain)
        self.X = X
        Ycentered = (Y[0], [Y[1][i] - self.Ymean_func(Y[0][i].squeeze()) for i in range(n)])
        coefsY = BiBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = KernelRidge(self.kernel, self.regu)
            reg.fit(X, coefsY[:, prob])
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew):
        preds = np.array([reg.predict(Xnew) for reg in self.regressors]).T
        return preds

    def predict_from_coefs(self, pred_coefs, yin_new):
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            return pred_coefs.dot(basis_evals.T) + np.expand_dims(self.Ymean_func(yin_new.squeeze()), axis=0)
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        pred_coefs = self.predict(Xnew)
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        return preds

