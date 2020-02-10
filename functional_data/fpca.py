import numpy as np

from functional_data import basis


class FunctionalPCA:

    def __init__(self, domain, n_evals, output_smoother):
        self.domain = domain
        self.output_smoother = output_smoother
        self.n_evals = n_evals
        self.eig_vals = None

    def fit(self, Xfuncs):
        n = len(Xfuncs)
        space = np.linspace(self.domain[0, 0], self.domain[0, 1], self.n_evals)
        data_mat = np.array([func(space) for func in Xfuncs]).squeeze()
        u, v, w = np.linalg.svd(data_mat, full_matrices=False)
        a = (self.domain[0, 1] - self.domain[0, 0]) / self.n_evals
        self.output_smoother.fit([space for i in range(n)], (1 / np.sqrt(a)) * w)
        self.eig_vals = v ** 2

    def get_principal_basis(self, k):
        return basis.BasisFromSmoothFunctions(self.output_smoother.get_functions()[:k], 1, self.domain)

    def get_eig_vals(self, k):
        return self.eig_vals[:k]

    def get_regressors(self, k):
        return self.output_smoother.get_functions()[:k]

    def predict(self, xlocs):
        return self.output_smoother.predict(xlocs)

