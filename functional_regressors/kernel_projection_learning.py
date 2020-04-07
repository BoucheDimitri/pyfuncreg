import numpy as np
from slycot import sb04qd

from functional_data import basis
from functional_regressors import regularization
from functional_data import discrete_functional_data as disc_fd
from functional_regressors.functional_regressor import FunctionalRegressor


class SeparableSubridgeSVD:
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
    def __init__(self, regu, kernel, B, phi_adj_phi):
        self.kernel = kernel
        self.B = B
        self.phi_adj_phi = phi_adj_phi
        self.K = None
        self.alpha = None
        self.X = None
        self.v, self.V = None, None
        self.u, self.U = None, None
        self.regu = regu

    def fit(self, X, Y, K=None):
        self.X = X
        if K is not None:
            self.K = K
        else:
            self.K = self.kernel(X, X)
        self.v, self.V = np.linalg.eigh(self.phi_adj_phi.dot(self.B))
        self.u, self.U = np.linalg.eigh(self.K)
        self.Ytilde = Y.dot(self.V)
        n = len(X)
        m = len(self.B)
        alpha_tilde = np.zeros((n, m))
        regus = self.regu * n / self.v
        # regus = self.regu / self.v
        for i in range(m):
            alpha_tilde[:, i] = self.U.dot(np.diag(1 / (self.u + regus[i]))).dot(self.U.T).dot(self.Ytilde[:, i]) / self.v[i]
        self.alpha = (self.V.dot(alpha_tilde.T)).T

    def predict(self, Xnew):
        Knew = self.kernel(self.X, Xnew)
        preds = (self.B.dot(self.alpha.T.dot(Knew.T))).T
        return preds


class SeparableSubridgeSVDPath:
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
    def __init__(self, kernel, B, phi_adj_phi):
        self.kernel = kernel
        self.B = B
        self.phi_adj_phi = phi_adj_phi
        self.K = None
        self.alpha = None
        self.X = None
        self.v, self.V = None, None
        self.u, self.U = None, None

    def fit(self, X, Y, K=None):
        self.X = X
        if K is not None:
            self.K = K
        else:
            self.K = self.kernel(X, X)
        self.v, self.V = np.linalg.eigh(self.phi_adj_phi.dot(self.B))
        self.u, self.U = np.linalg.eigh(self.K)
        self.Ytilde = Y.dot(self.V)

    def compute_alpha(self, regu):
        n = len(self.X)
        m = len(self.B)
        alpha_tilde = np.zeros((n, m))
        regus = regu * n / self.v
        # regus = self.regu / self.v
        for i in range(m):
            alpha_tilde[:, i] = self.U.dot(np.diag(1 / (self.u + regus[i]))).dot(self.U.T).dot(self.Ytilde[:, i]) / self.v[i]
        return (self.V.dot(alpha_tilde.T)).T

    def predict(self, Xnew, regu):
        Knew = self.kernel(self.X, Xnew)
        alpha = self.compute_alpha(regu)
        preds = (self.B.dot(alpha.T.dot(Knew.T))).T
        return preds


class SeparableSubridgeSylv:
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
    def __init__(self, regu, kernel, B, phi_adj_phi):
        self.kernel = kernel
        self.B = B
        self.phi_adj_phi = phi_adj_phi
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
        self.alpha = sb04qd(n, m, self.K / (self.regu * n), self.phi_adj_phi.dot(self.B), Y / (self.regu * n))

    def predict(self, Xnew):
        Knew = self.kernel(self.X, Xnew)
        preds = (self.B.dot(self.alpha.T.dot(Knew.T))).T
        return preds


class SeperableKPL(FunctionalRegressor):
    """
    Parameters
    ----------
    kernel : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns an array_like object with shape = [n_samples_x1, n_samples_x0].
    B : regularization.OutputMatrix or array_like or tuple
        Matrix encoding the similarities between output tasks
    basis_out : functional_data.basis.Basis or tuple
        The output dictionary of functions
    regu : float
        Regularization parameter
    center_output : bool
        Should output be centered
    """
    def __init__(self, regu, kernel, B, basis_out, center_output=False):
        super().__init__()
        self.kernel = kernel
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean_func = None
        # If a basis is given, the output dictionary is fixed, else it is generated from the passed config upon fitting
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        # If a numpy array is explicitly it remains fixed, else it is generated with the output_basis
        # upon fitting using the passed config
        self.B_abstract, self.B = regularization.set_output_matrix_config(B)
        # else:
        #     raise ValueError("B must be either numpy.ndarray or functional_regressors.regularization.OutputMatrix")
        # Attributes used for centering
        self.center_output = center_output
        # Underlying solver
        self.ovkridge = None

    def generate_output_basis(self, Y):
        if self.basis_out is None:
            self.basis_out = basis.generate_basis(*self.basis_out_config)
        if isinstance(self.basis_out, basis.DataDependantBasis):
            self.basis_out.fit(*Y)

    def generate_output_matrix(self):
        if self.B is None:
            if isinstance(self.B_abstract, regularization.OutputMatrix):
                self.B = self.B_abstract.get_matrix(self.basis_out)
            else:
                self.B = regularization.generate_output_matrix(
                    self.B_abstract[0], self.B_abstract[1]).get_matrix(self.basis_out)

    def fit(self, X, Y, K=None):
        # Center output functions if relevant
        # start_center = perf_counter()
        self.Ymean_func = disc_fd.mean_func(*Y)
        if self.center_output:
            Ycentered = disc_fd.center_discrete(*Y, self.Ymean_func)
            Ycentered = disc_fd.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd.to_discrete_general(*Y)
        # end_center = perf_counter()
        # print("Centering of the data perf :" + str(end_center - start_center))
        # Memorize training input data
        self.X = X
        # Generate output dictionary
        # start_basis = perf_counter()
        self.generate_output_basis(Ycentered)
        # end_basis = perf_counter()
        # print("Generating the output basis perf :" + str(end_basis - start_basis))
        # Generate output matrix
        # start_outmat = perf_counter()
        self.generate_output_matrix()
        # end_outmat = perf_counter()
        # print("Generating output matrix perf :" + str(end_outmat - start_outmat))
        # Compute input kernel matrix if not given
        # start_kmat = perf_counter()
        if K is None:
            K = self.kernel(X, X)
        # end_kmat = perf_counter()
        # print("Computing kernel matrix perf: " + str(end_kmat - start_kmat))
        n = K.shape[0]
        # Compute approximate dot product between output functions and dictionary functions
        # return Ycentered, self.output_basis
        # start_phimats = perf_counter()
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.basis_out.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        # end_phimats = perf_counter()
        # print("Computing phi mats perf: " + str(end_phimats - start_phimats))
        # start_yproj = perf_counter()
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # end_yproj = perf_counter()
        # print("Computing Y projection on dict perf: " + str(end_yproj - start_yproj))
        # return Yproj
        # Fit ovk ridge using those approximate projections
        # start_fitovk = perf_counter()
        phi_adj_phi = self.basis_out.get_gram_matrix()
        # phi_adj_phi = np.eye(self.basis_out.n_basis)
        self.ovkridge = SeparableSubridgeSylv(self.regu, self.kernel, self.B, phi_adj_phi)
        # self.ovkridge = SeparableSubridgeSVD(self.regu, self.kernel, self.B, phi_adj_phi)
        # return self.ovkridge, Yproj
        self.ovkridge.fit(X, Yproj, K=K)
        # end_fitovk = perf_counter()
        # print("Fitting the OVK Ridge: " + str(end_fitovk - start_fitovk))

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_from_coefs(self, pred_coefs, yin_new):
        if pred_coefs.ndim == 1:
            pred_coefs = np.expand_dims(pred_coefs, axis=0)
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output:
            mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
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


    # def predict_evaluate(self, Xnew, yin_new):
    #     pred_coefs = self.predict(Xnew)
    #     basis_evals = self.basis_out.compute_matrix(yin_new)
    #     if self.center_output is not False:
    #         mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
    #         return pred_coefs.dot(basis_evals.T) + mean_eval
    #     else:
    #         return pred_coefs.dot(basis_evals.T)
    #
    # def predict_evaluate_diff_locs(self, Xnew, Yins_new):
    #     n_preds = len(Xnew)
    #     preds = []
    #     for i in range(n_preds):
    #         preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i])))
    #     return preds


# TODO : finish this

class SeperableKPLRegpath():
    """
    Parameters
    ----------
    kernel : callable
        Must support being called on two array_like objects X0, X1. If len(X0) = n_samples0 and len(X1) = n_samples1,
        must returns an array_like object with shape = [n_samples_x1, n_samples_x0].
    B : regularization.OutputMatrix or array_like or tuple
        Matrix encoding the similarities between output tasks
    basis_out : functional_data.basis.Basis or tuple
        The output dictionary of functions
    regu : float
        Regularization parameter
    center_output : bool
        Should output be centered
    """
    def __init__(self, kernel, B, basis_out, center_output=False):
        self.kernel = kernel
        self.alpha = None
        self.X = None
        self.Ymean_func = None
        # If a basis is given, the output dictionary is fixed, else it is generated from the passed config upon fitting
        self.basis_out_config, self.basis_out = basis.set_basis_config(basis_out)
        # If a numpy array is explicitly it remains fixed, else it is generated with the output_basis
        # upon fitting using the passed config
        self.B_abstract, self.B = regularization.set_output_matrix_config(B)
        # else:
        #     raise ValueError("B must be either numpy.ndarray or functional_regressors.regularization.OutputMatrix")
        # Attributes used for centering
        self.center_output = center_output
        # Underlying solver
        self.ovkridge = None

    def generate_output_basis(self, Y):
        if self.basis_out is None:
            self.basis_out = basis.generate_basis(*self.basis_out_config)
        if isinstance(self.basis_out, basis.DataDependantBasis):
            self.basis_out.fit(*Y)

    def generate_output_matrix(self):
        if self.B is None:
            if isinstance(self.B_abstract, regularization.OutputMatrix):
                self.B = self.B_abstract.get_matrix(self.basis_out)
            else:
                self.B = regularization.generate_output_matrix(
                    self.B_abstract[0], self.B_abstract[1]).get_matrix(self.basis_out)

    def fit(self, X, Y, K=None):
        # Center output functions if relevant
        # start_center = perf_counter()
        self.Ymean_func = disc_fd.mean_func(*Y)
        if self.center_output:
            Ycentered = disc_fd.center_discrete(*Y, self.Ymean_func)
            Ycentered = disc_fd.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd.to_discrete_general(*Y)
        # end_center = perf_counter()
        # print("Centering of the data perf :" + str(end_center - start_center))
        # Memorize training input data
        self.X = X
        # Generate output dictionary
        # start_basis = perf_counter()
        self.generate_output_basis(Ycentered)
        # end_basis = perf_counter()
        # print("Generating the output basis perf :" + str(end_basis - start_basis))
        # Generate output matrix
        # start_outmat = perf_counter()
        self.generate_output_matrix()
        # end_outmat = perf_counter()
        # print("Generating output matrix perf :" + str(end_outmat - start_outmat))
        # Compute input kernel matrix if not given
        # start_kmat = perf_counter()
        if K is None:
            K = self.kernel(X, X)
        # end_kmat = perf_counter()
        # print("Computing kernel matrix perf: " + str(end_kmat - start_kmat))
        n = K.shape[0]
        # Compute approximate dot product between output functions and dictionary functions
        # return Ycentered, self.output_basis
        # start_phimats = perf_counter()
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.basis_out.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        # end_phimats = perf_counter()
        # print("Computing phi mats perf: " + str(end_phimats - start_phimats))
        # start_yproj = perf_counter()
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # end_yproj = perf_counter()
        # print("Computing Y projection on dict perf: " + str(end_yproj - start_yproj))
        # return Yproj
        # Fit ovk ridge using those approximate projections
        # start_fitovk = perf_counter()
        phi_adj_phi = self.basis_out.get_gram_matrix()
        # phi_adj_phi = np.eye(self.basis_out.n_basis)
        self.ovkridge = SeparableSubridgeSVDPath(self.kernel, self.B, phi_adj_phi)
        # self.ovkridge = SeparableSubridgeSVD(self.regu, self.kernel, self.B, phi_adj_phi)
        # return self.ovkridge, Yproj
        self.ovkridge.fit(X, Yproj, K=K)
        # end_fitovk = perf_counter()
        # print("Fitting the OVK Ridge: " + str(end_fitovk - start_fitovk))

    def predict(self, Xnew, regu):
        return self.ovkridge.predict(Xnew, regu)

    def predict_evaluate(self, Xnew, yin_new, regu):
        pred_coefs = self.predict(Xnew, regu)
        basis_evals = self.basis_out.compute_matrix(yin_new)
        if self.center_output is not False:
            mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, regu):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i], regu)))
        return preds
