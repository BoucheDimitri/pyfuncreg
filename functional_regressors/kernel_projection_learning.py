import numpy as np
import functools

from functional_regressors import ovkernel_ridge
from functional_data import basis
from functional_data import fpca
from functional_data import smoothing
from functional_data import sparsely_observed
from functional_data import functional_algebra
from functional_regressors import regularization
from functional_data import discrete_functional_data


class SperableKPL:
    """
    Parameters
    ----------
    kernel_scalar: functional_regressors.kernels.ScalarKernel
        The scalar kernel
    B: regularization.OutputMatrix or array-like, shape = [n_output_features, n_output_features]
        Matrix encoding the similarities between output tasks
    output_basis: functional_data.basis.Basis or tuple
        The output dictionary of functions
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    """
    def __init__(self, kernel_scalar, B, output_basis, regu, center_output="samelocs_missing"):
        self.kernel_scalar = kernel_scalar
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean = None
        # If a basis is given, the output dictionary is fixed, else it is generated from the passed config upon fitting
        if isinstance(output_basis, basis.Basis):
            self.output_basis = output_basis
            self.output_basis_config = None
        else:
            self.output_basis_config = output_basis
            self.output_basis = None
        # If a numpy array is explicitly it remains fixed, else it is generated with the output_basis
        # upon fitting using the passed config
        if isinstance(B, np.ndarray):
            self.B = B
            self.abstract_B = None
        elif isinstance(B, regularization.OutputMatrix):
            self.B = None
            self.abstract_B = B
        else:
            raise ValueError("B must be either numpy.ndarray or functional_regressors.regularization.OutputMatrix")
        # Attributes used for centering
        self.center_output = center_output
        # Underlying solver
        self.ovkridge = None

    def generate_output_basis(self, Y):
        if self.output_basis is None:
            #TODO: In the case where we want to penalize according to eigenvalues, how to we do so
            if basis.is_data_dependant(self.output_basis_config[0]):
                self.output_basis_config[1]["Y"] = Y
            self.output_basis = basis.generate_basis(self.output_basis_config[0], self.output_basis_config[1])

    def generate_output_matrix(self):
        if self.B is None:
            self.B = self.abstract_B.get_matrix(self.output_basis)

    def fit(self, X, Y, K=None):
        # Center output functions if relevant
        if self.center_output is not False:
            self.Ymean = discrete_functional_data.mean(Y[0], Y[1], mode=self.center_output)
            Ycentered = Y[0], discrete_functional_data.substract_function(Y[0], Y[1], self.Ymean)
        else:
            Ycentered = Y
        # Memorize training input data
        self.X = X
        # Generate output dictionary
        self.generate_output_basis(Y)
        # Generate output matrix
        self.generate_output_matrix()
        # Compute input kernel matrix if not given
        if K is None:
            K = self.kernel_scalar(X, X)
        n = K.shape[0]
        # Compute approximate dot product between output functions and dictionary functions
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # Fit ovk ridge using those approximate projections
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(self.kernel_scalar, self.B, self.regu)
        self.ovkridge.fit(X, Yproj)

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output is not False:
            mean_eval = np.expand_dims(self.Ymean(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i])))
        return preds


class KPLExact:
    """
    Parameters
    ----------
    kernel_scalar: functional_regressors.kernels.ScalarKernel
        The scalar kernel
    B: array-like, shape = [n_output_features, n_output_features]
        Matrix encoding the similarities between output tasks
    output_basis: functional_data.basis.Basis
        The output dictionary of functions
    regu: float
        Regularization parameter
    center_output: bool, optional
        Should outputs be centered ?
    non_padded_index: array-like, optional
        The full set of locations of observations, only needed if center_output is True
    """
    def __init__(self, kernel_scalar, B, output_basis, regu, center_output=False, non_padded_index=None):
        self.kernel_scalar = kernel_scalar
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean = None
        self.output_basis = output_basis
        self.B = B
        self.center_output = center_output
        self.non_padded_index = non_padded_index
        self.full_output_locs = None
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(kernel_scalar, B, regu)

    @staticmethod
    def get_func_outputs(Y):
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(Y[0], Y[1])
        return smoother_out.get_functions()

    def fit(self, X, Y, K=None):
        if self.center_output:
            full_output_locs, Ymean = sparsely_observed.mean_missing(Y[0], Y[1])
            self.Ymean = Ymean[self.non_padded_index[0]:self.non_padded_index[1]]
            self.full_output_locs = full_output_locs[self.non_padded_index[0]:self.non_padded_index[1]]
            Ycentered = sparsely_observed.substract_missing(full_output_locs, Ymean, Y[0], Y[1])
        else:
            Ycentered = Y
        self.X = X
        if K is None:
            K = self.kernel_scalar(X, X)
        n = K.shape[0]
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        self.ovkridge.fit(X, Yproj)

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(yin_new.squeeze(), self.full_output_locs, self.Ymean), axis=0)
            return pred_coefs.dot(basis_evals.T) + extrapolate_mean
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i])))
        return preds


class KPLApprox:
    """
    Kernel projection learning using approximate L-BFGS solver

    Parameters
    ----------
    kernel_scalar: functional_regressors.kernels.ScalarKernel
        The scalar kernel
    B: array-like, shape = [n_output_features, n_output_features]
        Matrix encoding the similarities between output tasks
    output_basis: functional_data.basis.Basis
        The output dictionary of functions
    regu: float
        Regularization parameter
    solver: solvers.first_order.ScipySolver
        Solver for optimization
    center_output: bool, optional
        Should outputs be centered ?
    non_padded_index: array-like, optional
        The full set of locations of observations, only needed if center_output is True
    """
    def __init__(self, kernel_scalar, B, output_basis, regu, solver, center_output=False, non_padded_index=None):
        self.output_basis = output_basis
        self.kernel_scalar = kernel_scalar
        self.B = B
        self.regu = regu
        self.solver = solver
        self.alpha = None
        self.X = None
        self.Ymean = None
        self.full_output_locs = None
        self.center_output = center_output
        self.non_padded_index = non_padded_index

    @staticmethod
    def primal_objective(K, B, phi_mats, youts, regu, alpha):
        """
        Evaluation of the primal objective

        Parameters
        ----------
        K : numpy.ndarray
            The kernel matrix, must have shape [n_samples, n_samples]
        B : numpy.ndarray
            Output matrix of the separable matrix valued kernel, must have shape [n_output_features, n_output_features]
        phi_mats : iterable
            List of matrix of evaluation of the basis functions at the input point phi_mats[i]
            has shape [n_evals_nth_function, n_output_features]
        youts : iterable
            List of numpy.ndarray, youts[i] has shape [n_evals_nth_function, n_output_features]
        regu : float
            Regularization parameter
        alpha : numpy.ndarray
            The representer theorem coefficients, must have shape [n_output_features, n_samples]

        Returns
        -------
        float
            Evaluation of the primal objective
        """
        n = K.shape[0]
        if alpha.ndim == 1:
            alpha = alpha.reshape((B.shape[0], n))
        BalphaK = B.dot(alpha).dot(K)
        pred_mats = [phi_mats[i].T.dot(BalphaK[:, i]) for i in range(n)]
        weighted_norms = np.array([(1 / phi_mats[i].shape[1])
                                   * np.linalg.norm(pred_mats[i] - youts[i]) ** 2 for i in range(n)])
        regu_term = np.sum(np.multiply(K, alpha.T.dot(B.dot(alpha))))
        return weighted_norms.mean() + (regu / 2) * regu_term

    def get_primal_objective(self, K, phi_mats, youts):
        return functools.partial(KPLApprox.primal_objective,
                                 K,
                                 self.B,
                                 phi_mats,
                                 youts,
                                 self.regu)

    @staticmethod
    def get_primal_objective_static(K, B, phi_mats, youts, regu):
        return functools.partial(KPLApprox.primal_objective, K, B, phi_mats, youts, regu)

    @staticmethod
    def primal_gradient(K, B, phi_mats, youts, regu, alpha):
        n = K.shape[0]
        flatind = False
        if alpha.ndim == 1:
            alpha = alpha.reshape((B.shape[0], n))
            flatind = True
        weighted_phis_youts = np.array([(1 / youts[i].shape[0]) * phi_mats[i].dot(youts[i]) for i in range(n)]).T
        first_term = - (2 / n) * B.dot(weighted_phis_youts.dot(K))
        weighted_phisT_phis = [(1 / phi_mats[i].shape[1]) * phi_mats[i].dot(phi_mats[i].T) for i in range(n)]
        BalphaK = B.dot(alpha.dot(K))
        fixed_mat_second_term = B.dot(np.array([weighted_phisT_phis[i].dot(BalphaK[:, i]) for i in range(n)]).T)
        second_term = (2 / n) * fixed_mat_second_term.dot(K)
        regu_term = regu * B.dot(alpha.dot(K))
        if flatind:
            return (first_term + second_term + regu_term).flatten()
        else:
            return first_term + second_term + regu_term

    @staticmethod
    def get_primal_gradient_static(K, B, phi_mats, youts, regu):
        return functools.partial(KPLApprox.primal_gradient, K, B, phi_mats, youts, regu)

    def get_primal_gradient(self, K, phi_mats, youts):
        return functools.partial(KPLApprox.primal_gradient,
                                 K,
                                 self.B,
                                 phi_mats,
                                 youts,
                                 self.regu)

    def fit(self, X, Y, K=None, phi_mats=None, alpha0=None):
        """
        """
        if self.center_output:
            full_output_locs, Ymean = sparsely_observed.mean_missing(Y[0], Y[1])
            self.Ymean = Ymean[self.non_padded_index[0]:self.non_padded_index[1]]
            self.full_output_locs = full_output_locs[self.non_padded_index[0]:self.non_padded_index[1]]
            Ycentered = sparsely_observed.substract_missing(full_output_locs, Ymean, Y[0], Y[1])
        else:
            Ycentered = Y
        self.X = X
        if K is None:
            K = self.kernel_scalar(X, X)
        if phi_mats is None:
            phi_mats = [self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(len(Ycentered[0]))]
        fun = self.get_primal_objective(K, phi_mats, Ycentered[1])
        jac = self.get_primal_gradient(K, phi_mats, Ycentered[1])
        if alpha0 is None:
            alpha0 = np.zeros((self.output_basis.n_basis, K.shape[0])).flatten()
        sol, rec = self.solver(fun, alpha0, jac)
        self.alpha = sol.x.reshape((self.B.shape[0], K.shape[0]))
        return sol, rec

    def predict(self, Xnew):
        Knew = self.kernel_scalar(self.X, Xnew)
        return (self.B.dot(self.alpha.dot(Knew.T))).T

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output:
            extrapolate_mean = np.expand_dims(np.interp(yin_new.squeeze(), self.full_output_locs, self.Ymean), axis=0)
            return pred_coefs.dot(basis_evals.T) + extrapolate_mean
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate(Xnew[i], Yins_new[i])))
        return preds


class KPLExactFPCA:

    def __init__(self, kernel_scalar, regu, n_fpca, nevals_fpca=500, penalize_eigvals=0,
                 penalize_pow=1, center_output=True):
        """
        Kernel projection learning with FPCA dictionary using exact solver

        Parameters
        ----------
        kernel_scalar
        regu
        n_fpca
        nevals_fpca
        penalize_eigvals
        penalize_pow
        center_output
        """
        self.kernel_scalar = kernel_scalar
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean_func = None
        self.output_basis = None
        self.fpca = None
        self.nevals_fpca = nevals_fpca
        self.n_fpca = n_fpca
        self.penalize_eigvals = penalize_eigvals
        self.penalize_pow = penalize_pow
        self.B = None
        self.center_output = center_output
        self.ovkridge = None

    @staticmethod
    def get_func_outputs(Y):
        smoother_out = smoothing.LinearInterpSmoother()
        smoother_out.fit(Y[0], Y[1])
        return smoother_out.get_functions()

    def intialize_dict_and_B(self, Yfunc, domain):
        self.fpca = fpca.FunctionalPCA(domain, self.nevals_fpca, smoothing.LinearInterpSmoother())
        self.fpca.fit(Yfunc)
        if self.center_output:
            self.output_basis = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.n_fpca), 1, domain)
        else:
            self.output_basis = basis.BasisFromSmoothFunctions(self.fpca.get_regressors(self.n_fpca), 1, domain,
                                                               add_constant=True)
        if self.penalize_eigvals != 0 and self.penalize_pow == 1:
            eigvals = self.fpca.get_eig_vals(self.n_fpca)
            eigvals_norm = eigvals / np.max(eigvals)
            if self.center_output:
                self.B = np.diag(list(eigvals_norm ** self.penalize_eigvals / np.max(eigvals_norm ** self.penalize_eigvals)))
            else:
                self.B = np.diag(list(eigvals_norm ** self.penalize_eigvals / np.max(eigvals_norm ** self.penalize_eigvals)) + [1])
        elif self.penalize_pow != 1 and self.penalize_eigvals == 0:
            pows = np.arange(self.n_fpca)
            if self.center_output:
                self.B = np.diag(list(1 / self.penalize_pow ** pows))
            else:
                self.B = np.diag(list(1 / self.penalize_pow ** pows) + [1])
        else:
            if self.center_output:
                self.B = np.eye(self.n_fpca)
            else:
                self.B = np.diag(self.n_fpca + 1)

    def fit(self, X, Y, K=None):
        Yfunc = KPLExactFPCA.get_func_outputs(Y)
        self.Ymean_func = functional_algebra.mean_function(Yfunc)
        if self.center_output:
            Yfunc_centered = functional_algebra.diff_function_list(Yfunc, self.Ymean_func)
        else:
            Yfunc_centered = Yfunc
        domain = np.array([[Y[0][0][0], Y[0][0][-1]]])
        self.intialize_dict_and_B(Yfunc_centered, domain)
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(self.kernel_scalar, self.B, self.regu)
        n = len(Yfunc)
        Ycentered = (Y[0], [Y[1][i] - self.Ymean_func(Y[0][i].squeeze()) for i in range(n)])
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # return Yproj
        self.ovkridge.fit(X, Yproj)

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output:
            return pred_coefs.dot(basis_evals.T) + np.expand_dims(self.Ymean_func(yin_new.squeeze()), axis=0)
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i])))
        return preds