import numpy as np

from functional_regressors import ovkernel_ridge
from functional_data import basis
from functional_regressors import regularization
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors.functional_regressor import FunctionalRegressor


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
        self.Ymean_func = disc_fd1.mean_func(*Y)
        if self.center_output:
            Ycentered = disc_fd1.center_discrete(*Y, self.Ymean_func)
            Ycentered = disc_fd1.to_discrete_general(*Ycentered)
        else:
            Ycentered = disc_fd1.to_discrete_general(*Y)
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
        gram_dict = self.basis_out.gram_matrix()
        self.ovkridge = ovkernel_ridge.SeparableOVKRidge(self.regu, self.kernel, gram_dict.dot(self.B))
        self.ovkridge.fit(X, Yproj, K=K)
        # end_fitovk = perf_counter()
        # print("Fitting the OVK Ridge: " + str(end_fitovk - start_fitovk))

    def predict(self, Xnew):
        return self.ovkridge.predict(Xnew)

    def predict_evaluate(self, Xnew, yin_new):
        pred_coefs = self.predict(Xnew)
        basis_evals = self.basis_out.compute_matrix(yin_new)
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
