import numpy as np
import functools
from time import perf_counter

from functional_regressors import ovkernel_ridge
from functional_data import basis
from functional_data import fpca
from functional_data import smoothing
from functional_data import sparsely_observed
from functional_data import functional_algebra
from functional_regressors import regularization
from functional_data import discrete_functional_data
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
            self.B_abstract = None
        # elif isinstance(B, regularization.OutputMatrix):
        else:
            self.B = None
            self.B_abstract = B
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

    def fit(self, X, Y, K=None, mode='discrete_samelocs_regular_1d'):
        # Center output functions if relevant
        # start_center = perf_counter()
        Ywrapped = discrete_functional_data.wrap_functional_data(Y, mode)
        # Memorize mean function before signal extension
        if self.center_output:
            self.Ymean_func = Ywrapped.mean_func()
        # Extends the signal if relevant
        Ywrapped_extended = Ywrapped.extended_version(self.signal_ext[0], self.signal_ext[1])
        # Center with extended signal if relevant
        if self.center_output:
            Ycentered = Ywrapped_extended.centered_discrete_general()
        else:
            Ycentered = Ywrapped_extended.discrete_general()
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

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.output_basis.compute_matrix(yin_new)
        if self.center_output is not False:
            mean_eval = np.expand_dims(self.Ymean_func(yin_new), axis=0)
            return pred_coefs.dot(basis_evals.T) + mean_eval
        else:
            return pred_coefs.dot(basis_evals.T)

    def predict_evaluate_diff_locs(self, Xnew, Yins_new):
        n_preds = len(Xnew)
        preds = []
        for i in range(n_preds):
            preds.append(np.squeeze(self.predict_evaluate([Xnew[i]], Yins_new[i])))
        return preds
