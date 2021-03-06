import numpy as np
from sklearn import kernel_ridge
import time

from ..functional_data import basis, discrete_functional_data as disc_fd


def projection_coefs(X, func_basis, domain):
    n_samples = len(X[0])
    eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
    scalar_prods = np.array([eval_mats[i].T.dot(((domain[0, 1] - domain[0, 0])/X[1][i].shape[0]) * X[1][i])
                             for i in range(n_samples)])
    return scalar_prods


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
    regu : float
        Regularization parameter
    basis_in :
        The input orthonormal basis
    basis_rffs :
        Random Fourier Features used for the approximation
    basis_out :
        The output orthonormal basis
    center_output : bool, optional
        Should outputs be centered
    """
    def __init__(self, regu, basis_in, basis_rffs, basis_out, center_output=False):
        self.basis_in = basis_in
        # If a basis is given (both for input and output) else it is generated from the passed config upon fitting
        self.basis_in_config, self.basis_in = basis.set_basis_config(basis_in)
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        self.basis_rffs_config, self.basis_rffs = basis.set_basis_config(basis_rffs)
        self.ridge_inv = None
        self.regu = regu
        self.regressors = None
        self.center_output = center_output
        self.Ymean = None

    def generate_bases(self, X, Y):
        if self.basis_in is None:
            self.basis_in = basis.generate_basis(self.basis_in_config[0], self.basis_in_config[1])
        if isinstance(self.basis_in, basis.DataDependantBasis):
            self.basis_in.fit(X[0], X[1])
        if self.basis_out is None:
            self.basis_out = basis.generate_basis(self.basis_out_config[0], self.basis_out_config[1])
        if isinstance(self.basis_out, basis.DataDependantBasis):
            self.basis_in.fit(Y[0], Y[1])
        if self.basis_rffs is None:
            self.basis_rffs_config[1]["input_dim"] = self.basis_in.n_basis
            self.basis_rffs = basis.generate_basis(self.basis_rffs_config[0],
                                                   {**self.basis_rffs_config[1], **{"compute_gram": False}})

    def fit(self, X, Y, return_cputime=False):
        self.Ymean = disc_fd.mean_func(*Y)
        X_dg = disc_fd.to_discrete_general(*X)
        Y_dg = disc_fd.to_discrete_general(*Y)
        Ycentered = disc_fd.center_discrete(*Y_dg, self.Ymean)
        self.generate_bases(X_dg, Ycentered)
        coefsX = projection_coefs(X_dg, self.basis_in, self.basis_in.domain)
        coefsY = projection_coefs(Ycentered, self.basis_out, self.basis_out.domain)
        n_probs = coefsY.shape[1]
        # regressors = []
        # for prob in range(n_probs):
        #     reg = smoothing.ExpandedRidge(self.regu, self.basis_rffs)
        #     reg.fit(coefsX, coefsY[:, prob])
        #     regressors.append(reg)
        # self.regressors = regressors
        Z = self.basis_rffs.compute_matrix(coefsX)
        ridge_mat = Z.T.dot(Z) + self.regu * np.eye(self.basis_rffs.n_basis)
        start = time.process_time()
        self.ridge_inv = np.linalg.inv(ridge_mat)
        dual_coefs = []
        for prob in range(n_probs):
            dual_coefs.append(self.ridge_inv.dot(Z.T.dot(coefsY[:, prob])))
        self.dual_coefs = dual_coefs
        end = time.process_time()
        if return_cputime:
            return end - start

    def predict(self, Xnew):
        Xnew_dg = disc_fd.to_discrete_general(*Xnew)
        coefsXnew = projection_coefs(Xnew_dg, self.basis_in, self.basis_in.domain)
        Znew = self.basis_rffs.compute_matrix(coefsXnew)
        # preds = np.array([reg(coefsXnew) for reg in self.regressors]).T
        preds = np.array([Znew.dot(alpha) for alpha in self.dual_coefs]).T
        return preds

    def predict_from_coefs(self, pred_coefs, yin_new):
        if pred_coefs.ndim == 1:
            pred_coefs = np.expand_dims(pred_coefs, axis=0)
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            mean_eval = np.expand_dims(self.Ymean(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew[1])
        preds = []
        pred_coefs = self.predict(Xnew)
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        return preds


class BiBasisEstimator:
    """
    Triple basis estimator with structured input

    Parameters
    ----------
    kernel : functional_regressors.kernels.ScalarKernel
        The input kernel
    basis_out: functional_data.basis.Basis
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    """
    def __init__(self, regu, kernel, basis_out, center_output=False):
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        self.kernel = kernel
        self.regu = regu
        # self.regressors = None
        self.dual_coefs = None
        self.KnlambdaI_inv = None
        self.X = None
        self.Ymean = None
        self.center_output = center_output

    def generate_base(self, Y):
        if self.basis_out is None:
            self.basis_out = basis.generate_basis(self.basis_out_config[0], self.basis_out_config[1])
        if isinstance(self.basis_out, basis.DataDependantBasis):
            self.basis_out.fit(*Y)

    def fit(self, X, Y, K=None, return_cputime=False):
        self.X = X
        self.Ymean = disc_fd.mean_func(*Y)
        if self.center_output:
            Ycentered = disc_fd.center_discrete(*Y, self.Ymean)
            Ycentered = disc_fd.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd.to_discrete_general(*Y)
        self.generate_base(Ycentered)
        coefsY = projection_coefs(Ycentered, self.basis_out, self.basis_out.domain)
        if K is None:
            K = self.kernel(X, X)
        n_probs = coefsY.shape[1]
        start = time.process_time()
        self.KnlambdaI_inv = np.linalg.inv(K + K.shape[0] * self.regu * np.eye(K.shape[0]))
        dual_coefs = []
        for prob in range(n_probs):
            dual_coefs.append(self.KnlambdaI_inv.dot(coefsY[:, prob]))
        self.dual_coefs = dual_coefs
        end = time.process_time()
        if return_cputime:
            return end - start
        # start = time.process_time()
        # regressors = []
        # for prob in range(n_probs):
        #     reg = KernelRidge(self.kernel, self.regu)
        #     reg.fit(X, coefsY[:, prob], K=K)
        #     regressors.append(reg)
        # self.regressors = regressors
        # end = time.process_time()
        # if return_cputime:
        #     return end - start

    def predict(self, Xnew):
        # preds = np.array([reg.predict(Xnew) for reg in self.regressors]).T
        Knew = self.kernel(self.X, Xnew)
        # preds = np.array([Knew.dot(alpha) for alpha in self.dual_coefs]).T
        preds = np.array(self.dual_coefs).dot(Knew.T).T
        # preds = np
        return preds

    def predict_from_coefs(self, pred_coefs, yin_new):
        if pred_coefs.ndim == 1:
            pred_coefs = np.expand_dims(pred_coefs, axis=0)
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            mean_eval = np.expand_dims(self.Ymean(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, return_cputime=False):
        start = time.process_time()
        n_preds = len(Xnew)
        preds = []
        pred_coefs = self.predict(Xnew)
        # end = time.process_time()
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        end = time.process_time()
        if return_cputime:
            return preds, end - start
        else:
            return preds
