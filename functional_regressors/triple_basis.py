import numpy as np
from sklearn import kernel_ridge

from functional_data import basis
from functional_data.DEPRECATED import discrete_functional_data as disc_fd
from functional_data import smoothing
from functional_data import discrete_functional_data as disc_fd1


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
    basis_in:
        The input orthonormal basis
    basis_rffs:
        Random Fourier Features used for the approximation
    basis_out:
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered
    """
    def __init__(self, basis_in, basis_rffs, basis_out, regu, center_output=False,
                 signal_ext_input=None, signal_ext_output=None):
        self.basis_in = basis_in
        # If a basis is given (both for input and outpu) else it is generated from the passed config upon fitting
        self.basis_in_config, self.basis_in = basis.set_basis_config(basis_in)
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        self.basis_rffs_config, self.basis_rffs = basis.set_basis_config(basis_rffs)
        self.regu = regu
        self.regressors = None
        self.center_output = center_output
        self.Ymean = None
        self.signal_ext_input = signal_ext_input
        self.signal_ext_output = signal_ext_output

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        scalar_prods = np.array([eval_mats[i].T.dot((1/X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

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
            self.basis_rffs = basis.generate_basis(self.basis_rffs_config[0], self.basis_rffs_config[1])

    def fit(self, X, Y, input_data_format="discrete_samelocs_regular_1d",
            output_data_format="discrete_samelocs_regular_1d"):
        _, X_dg = disc_fd.preprocess_data(X, self.signal_ext_input, False, input_data_format)
        self.Ymean, Ycentered = disc_fd.preprocess_data(
            Y, self.signal_ext_output, self.center_output, output_data_format)
        self.generate_bases(X_dg, Ycentered)
        coefsX = TripleBasisEstimator.projection_coefs(X_dg, self.basis_in)
        coefsY = TripleBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = smoothing.ExpandedRidge(self.regu, self.basis_rffs)
            reg.fit(coefsX, coefsY[:, prob])
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew, input_data_format="discrete_samelocs_regular_1d"):
        Xnew_dg = disc_fd.to_discrete_general(Xnew, input_data_format)
        coefsXnew = TripleBasisEstimator.projection_coefs(Xnew_dg, self.basis_in)
        preds = np.array([reg(coefsXnew) for reg in self.regressors]).T
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

    def predict_evaluate(self, Xnew, yin_new, input_data_format):
        pred_coefs = self.predict(Xnew, input_data_format)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, input_data_format):
        n_preds = len(Xnew[1])
        preds = []
        pred_coefs = self.predict(Xnew, input_data_format)
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        return preds


class TripleBasisEstimatorBis:
    """
    Triple basis estimator

    Parameters
    ----------
    basis_in:
        The input orthonormal basis
    basis_rffs:
        Random Fourier Features used for the approximation
    basis_out:
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered
    """
    def __init__(self, basis_in, basis_rffs, basis_out, regu, center_output=False):
        self.basis_in = basis_in
        # If a basis is given (both for input and outpu) else it is generated from the passed config upon fitting
        self.basis_in_config, self.basis_in = basis.set_basis_config(basis_in)
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        self.basis_rffs_config, self.basis_rffs = basis.set_basis_config(basis_rffs)
        self.regu = regu
        self.regressors = None
        self.center_output = center_output
        self.Ymean = None

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        scalar_prods = np.array([eval_mats[i].T.dot((1/X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

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
            self.basis_rffs = basis.generate_basis(self.basis_rffs_config[0], self.basis_rffs_config[1])

    def fit(self, X, Y):
        self.Ymean = disc_fd1.mean_func(*Y)
        X_dg = disc_fd1.to_discrete_general(*X)
        Y_dg = disc_fd1.to_discrete_general(*Y)
        Ycentered = disc_fd1.center_discrete(*Y_dg, self.Ymean)
        self.generate_bases(X_dg, Ycentered)
        coefsX = TripleBasisEstimator.projection_coefs(X_dg, self.basis_in)
        coefsY = TripleBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        # return coefsX, coefsY
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = smoothing.ExpandedRidge(self.regu, self.basis_rffs)
            reg.fit(coefsX, coefsY[:, prob])
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew):
        Xnew_dg = disc_fd1.to_discrete_general(*Xnew)
        coefsXnew = TripleBasisEstimator.projection_coefs(Xnew_dg, self.basis_in)
        preds = np.array([reg(coefsXnew) for reg in self.regressors]).T
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
    kernel: functional_regressors.kernels.ScalarKernel
        The input kernel
    basis_out: functional_data.basis.Basis
        The output orthonormal basis
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    """
    def __init__(self, kernel, basis_out, regu, center_output=False, signal_ext=None):
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        self.kernel = kernel
        self.regu = regu
        self.regressors = None
        self.X = None
        self.Ymean = None
        self.signal_ext = signal_ext
        self.center_output = center_output

    @staticmethod
    def projection_coefs(X, func_basis):
        n_samples = len(X[0])
        eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
        scalar_prods = np.array([eval_mats[i].T.dot((1 / X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
        return scalar_prods

    def generate_base(self, Y):
        if self.basis_out is None:
            self.basis_out = basis.generate_basis(self.basis_out_config[0], self.basis_out_config[1])
        if isinstance(self.basis_out, basis.DataDependantBasis):
            self.basis_out.fit(Y[0], Y[1])

    def fit(self, X, Y, K=None, input_data_format="vector", output_data_format="discrete_samelocs_regular_1d"):
        self.Ymean, Ycentered = disc_fd.preprocess_data(
            Y, self.signal_ext, self.center_output, output_data_format)
        self.X = X
        self.generate_base(Ycentered)
        coefsY = BiBasisEstimator.projection_coefs(Ycentered, self.basis_out)
        if K is None:
            K = self.kernel(X, X)
        n_probs = coefsY.shape[1]
        regressors = []
        for prob in range(n_probs):
            reg = KernelRidge(self.kernel, self.regu)
            reg.fit(X, coefsY[:, prob], K=K)
            regressors.append(reg)
        self.regressors = regressors

    def predict(self, Xnew, input_data_format="vector"):
        preds = np.array([reg.predict(Xnew) for reg in self.regressors]).T
        return preds

    def predict_from_coefs(self, pred_coefs, yin_new):
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(yin_new.squeeze(), self.Ymean[0], self.Ymean[1]), axis=0)
            return pred_coefs.dot(basis_evals.T) + extrapolate_mean
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate(self, Xnew, yin_new, input_data_format="vector"):
        pred_coefs = self.predict(Xnew, input_data_format)
        return self.predict_from_coefs(pred_coefs, yin_new)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, input_data_format="vector"):
        n_preds = len(Xnew)
        preds = []
        pred_coefs = self.predict(Xnew, input_data_format)
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
        return preds


# class BiBasisEstimatorFpca:
#     """
#     Triple basis estimator with structured input and FPCA output basis
#
#     Parameters
#     ----------
#     kernel: functional_regressors.kernels.ScalarKernel
#         The input kernel
#     regu: float
#         Regularization parameter
#     nfpca: int
#         The number of function principal components to include
#     nevals_fpca: int
#         The number of evaluations to use for approximation of the FPCA
#     center_output: bool, optional
#         Should outputs be centered ?
#     """
#     def __init__(self, kernel, regu, nfpca, nevals_fpca=500, center_output=True):
#         self.kernel = kernel
#         self.regu = regu
#         self.nfpca = nfpca
#         self.fpca = None
#         self.Ymean_func = None
#         self.regressors = None
#         self.X = None
#         self.basis_out = None
#         self.Ymean = None
#         self.center_output = center_output
#         self.nevals_fpca = nevals_fpca
#
#     @staticmethod
#     def projection_coefs(X, func_basis):
#         n_samples = len(X[0])
#         eval_mats = [func_basis.compute_matrix(X[0][i]) for i in range(n_samples)]
#         # shape = (n_samples, n_basis)
#         scalar_prods = np.array([eval_mats[i].T.dot((1 / X[1][i].shape[0]) * X[1][i]) for i in range(n_samples)])
#         return scalar_prods
#
#     @staticmethod
#     def get_func_outputs(Y):
#         smoother_out = smoothing.LinearInterpSmoother()
#         smoother_out.fit(Y[0], Y[1])
#         return smoother_out.get_functions()
#
#     def intialize_dict(self, Yfunc, domain):
#         self.fpca = fpca.FunctionalPCA(domain, self.nevals_fpca, smoothing.LinearInterpSmoother())
#         self.fpca.fit(Yfunc)
#         if self.center_output:
#             self.basis_out = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.nfpca), 1, domain)
#         else:
#             self.basis_out = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.nfpca), 1, domain,
#                                                             add_constant=True)
#
#     def fit(self, X, Y, K=None):
#         n = len(X)
#         Yfunc = BiBasisEstimatorFpca.get_func_outputs(Y)
#         self.Ymean_func = functional_algebra.mean_function(Yfunc)
#         if self.center_output:
#             Yfunc_centered = functional_algebra.diff_function_list(Yfunc, self.Ymean_func)
#         else:
#             Yfunc_centered = Yfunc
#         domain = np.array([[Y[0][0][0], Y[0][0][-1]]])
#         self.intialize_dict(Yfunc_centered, domain)
#         self.X = X
#         Ycentered = (Y[0], [Y[1][i] - self.Ymean_func(Y[0][i].squeeze()) for i in range(n)])
#         coefsY = BiBasisEstimator.projection_coefs(Ycentered, self.basis_out)
#         n_probs = coefsY.shape[1]
#         regressors = []
#         for prob in range(n_probs):
#             reg = KernelRidge(self.kernel, self.regu)
#             reg.fit(X, coefsY[:, prob])
#             regressors.append(reg)
#         self.regressors = regressors
#
#     def predict(self, Xnew):
#         preds = np.array([reg.predict(Xnew) for reg in self.regressors]).T
#         return preds
#
#     def predict_from_coefs(self, pred_coefs, yin_new):
#         basis_evals = self.basis_out.compute_matrix(yin_new)
#         if self.center_output:
#             return pred_coefs.dot(basis_evals.T) + np.expand_dims(self.Ymean_func(yin_new.squeeze()), axis=0)
#         else:
#             return pred_coefs.dot(basis_evals.T)
#
#     def predict_evaluate(self, Xnew, yin_new):
#         pred_coefs = self.predict(Xnew)
#         return self.predict_from_coefs(pred_coefs, yin_new)
#
#     def predict_evaluate_diff_locs(self, Xnew, Yins_new):
#         n_preds = len(Xnew)
#         preds = []
#         pred_coefs = self.predict(Xnew)
#         for i in range(n_preds):
#             preds.append(np.squeeze(self.predict_from_coefs(pred_coefs[i], Yins_new[i])))
#         return preds

