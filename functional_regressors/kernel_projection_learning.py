import numpy as np

from functional_regressors import ovkernel_ridge
from functional_data import basis
from functional_regressors import regularization
from functional_data.DEPRECATED import discrete_functional_data
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors.functional_regressor import FunctionalRegressor


class SeperableKPL(FunctionalRegressor):
    """
    Parameters
    ----------
    kernel_scalar : functional_regressors.kernels.ScalarKernel
        The scalar kernel
    B : regularization.OutputMatrix or array_like or tuple
        Matrix encoding the similarities between output tasks
    output_basis : functional_data.basis.Basis or tuple
        The output dictionary of functions
    regu : float
        Regularization parameter
    center_output : bool
        Should output be centered
    """
    def __init__(self, kernel_scalar, B, output_basis, regu, center_output=False, signal_ext=None):
        super().__init__()
        self.kernel_scalar = kernel_scalar
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean_func = None
        # If a basis is given, the output dictionary is fixed, else it is generated from the passed config upon fitting
        self.output_basis_config, self.output_basis = basis.set_basis_config(output_basis)
        # If a numpy array is explicitly it remains fixed, else it is generated with the output_basis
        # upon fitting using the passed config
        self.B_abstract, self.B = regularization.set_output_matrix_config(B)
        # else:
        #     raise ValueError("B must be either numpy.ndarray or functional_regressors.regularization.OutputMatrix")
        # Attributes used for centering
        self.center_output = center_output
        # Signal extension parameters
        self.signal_ext = signal_ext
        # Underlying solver
        self.ovkridge = None

    def generate_output_basis(self, Y):
        if self.output_basis is None:
            self.output_basis = basis.generate_basis(self.output_basis_config[0], self.output_basis_config[1])
        if isinstance(self.output_basis, basis.DataDependantBasis):
            self.output_basis.fit(Y[0], Y[1])

    def generate_output_matrix(self):
        if self.B is None:
            if isinstance(self.B_abstract, regularization.OutputMatrix):
                self.B = self.B_abstract.get_matrix(self.output_basis)
            else:
                self.B = regularization.generate_output_matrix(
                    self.B_abstract[0], self.B_abstract[1]).get_matrix(self.output_basis)

    def fit(self, X, Y, K=None, input_data_format="vector", output_data_format='discrete_samelocs_regular_1d'):
        # Center output functions if relevant
        # start_center = perf_counter()
        self.Ymean, Ycentered = discrete_functional_data.preprocess_data(Y, self.signal_ext,
                                                                         self.center_output,
                                                                         output_data_format)
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
            K = self.kernel_scalar(X, X)
        # end_kmat = perf_counter()
        # print("Computing kernel matrix perf: " + str(end_kmat - start_kmat))
        n = K.shape[0]
        # Compute approximate dot product between output functions and dictionary functions
        # return Ycentered, self.output_basis
        # start_phimats = perf_counter()
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        # end_phimats = perf_counter()
        # print("Computing phi mats perf: " + str(end_phimats - start_phimats))
        # start_yproj = perf_counter()
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # end_yproj = perf_counter()
        # print("Computing Y projection on dict perf: " + str(end_yproj - start_yproj))
        # return Yproj
        # Fit ovk ridge using those approximate projections
        # start_fitovk = perf_counter()
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(self.kernel_scalar, self.B, self.regu)
        self.ovkridge.fit(X, Yproj, K=K)
        # end_fitovk = perf_counter()
        # print("Fitting the OVK Ridge: " + str(end_fitovk - start_fitovk))

    def predict(self, Xnew, input_data_format="vector"):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new, input_data_format="vector"):
        pred_coefs = self.predict(Xnew, input_data_format)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output is not False:
            mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, input_data_format="vector"):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i], input_data_format)))
        return preds


class SeperableKPLBis(FunctionalRegressor):
    """
    Parameters
    ----------
    kernel_scalar : functional_regressors.kernels.ScalarKernel
        The scalar kernel
    B : regularization.OutputMatrix or array_like or tuple
        Matrix encoding the similarities between output tasks
    output_basis : functional_data.basis.Basis or tuple
        The output dictionary of functions
    regu : float
        Regularization parameter
    center_output : bool
        Should output be centered
    """
    def __init__(self, kernel_scalar, B, output_basis, regu, center_output=False):
        super().__init__()
        self.kernel_scalar = kernel_scalar
        self.regu = regu
        self.alpha = None
        self.X = None
        self.Ymean_func = None
        # If a basis is given, the output dictionary is fixed, else it is generated from the passed config upon fitting
        self.output_basis_config, self.output_basis = basis.set_basis_config(output_basis)
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
        if self.output_basis is None:
            self.output_basis = basis.generate_basis(self.output_basis_config[0], self.output_basis_config[1])
        if isinstance(self.output_basis, basis.DataDependantBasis):
            self.output_basis.fit(Y[0], Y[1])

    def generate_output_matrix(self):
        if self.B is None:
            if isinstance(self.B_abstract, regularization.OutputMatrix):
                self.B = self.B_abstract.get_matrix(self.output_basis)
            else:
                self.B = regularization.generate_output_matrix(
                    self.B_abstract[0], self.B_abstract[1]).get_matrix(self.output_basis)

    def fit(self, X, Y, K=None, input_data_format="vector"):
        # Center output functions if relevant
        # start_center = perf_counter()
        self.Ymean_func = disc_fd1.mean_func(Y[0], Y[1])
        if self.center_output:
            Ycentered = disc_fd1.center_discrete(Y[0], Y[1], self.Ymean_func)
            Ycentered = disc_fd1.to_discrete_general(Ycentered[0], Ycentered[1])
        else:
            Ycentered = disc_fd1.to_discrete_general(Y[0], Y[1])
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
            K = self.kernel_scalar(X, X)
        # end_kmat = perf_counter()
        # print("Computing kernel matrix perf: " + str(end_kmat - start_kmat))
        n = K.shape[0]
        # Compute approximate dot product between output functions and dictionary functions
        # return Ycentered, self.output_basis
        # start_phimats = perf_counter()
        phi_mats = [(1 / len(Ycentered[1][i]))
                    * self.output_basis.compute_matrix(Ycentered[0][i]).T for i in range(n)]
        # end_phimats = perf_counter()
        # print("Computing phi mats perf: " + str(end_phimats - start_phimats))
        # start_yproj = perf_counter()
        Yproj = np.array([phi_mats[i].dot(Ycentered[1][i]) for i in range(n)])
        # end_yproj = perf_counter()
        # print("Computing Y projection on dict perf: " + str(end_yproj - start_yproj))
        # return Yproj
        # Fit ovk ridge using those approximate projections
        # start_fitovk = perf_counter()
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(self.kernel_scalar, self.B, self.regu)
        self.ovkridge.fit(X, Yproj, K=K)
        # end_fitovk = perf_counter()
        # print("Fitting the OVK Ridge: " + str(end_fitovk - start_fitovk))

    def predict(self, Xnew, input_data_format="vector"):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new, input_data_format="vector"):
        pred_coefs = self.predict(Xnew, input_data_format)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output is not False:
            mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, input_data_format="vector"):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i], input_data_format)))
        return preds
