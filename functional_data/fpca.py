import numpy as np

from functional_data import basis
from functional_data import smoothing


class FunctionalPCA:
    """
    Parameters
    ----------
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain to consider
    n_evals: int
        number of evaluations to use for the discrete approximation
    output_smoother: functional_data.smoothing.Smoother
        The smoother to use to pass from the discrete approximation to functions
    """
    def __init__(self, domain, n_evals, output_smoother):
        self.domain = domain
        self.output_smoother = output_smoother
        self.n_evals = n_evals
        self.eig_vals = None

    def fit_from_funcs(self, Xfuncs):
        """
        Fit the PCA

        Parameters
        ----------
        Xfuncs: iterable of callables
            The functional data
        """
        n = len(Xfuncs)
        space = np.linspace(self.domain[0, 0], self.domain[0, 1], self.n_evals)
        data_mat = np.array([func(space) for func in Xfuncs]).squeeze()
        u, v, w = np.linalg.svd(data_mat, full_matrices=False)
        a = (self.domain[0, 1] - self.domain[0, 0]) / self.n_evals
        self.output_smoother.fit([space for i in range(n)], (1 / np.sqrt(a)) * w)
        self.eig_vals = v ** 2

    def fit_from_discrete_funcs(self, Xlocs, Xobs, smoother_in=smoothing.LinearInterpSmoother()):
        n = len(Xobs)
        smoother_in.fit(Xlocs, Xobs)
        Xfuncs = smoother_in.get_functions()
        space = np.linspace(self.domain[0, 0], self.domain[0, 1], self.n_evals)
        data_mat = np.array([func(space) for func in Xfuncs]).squeeze()
        u, v, w = np.linalg.svd(data_mat, full_matrices=False)
        a = (self.domain[0, 1] - self.domain[0, 0]) / self.n_evals
        self.output_smoother.fit([space for i in range(n)], (1 / np.sqrt(a)) * w)
        self.eig_vals = v ** 2

    def fit(self, *args):
        if


    def get_principal_basis(self, k):
        """
        Parameters
        ----------
        k: int
            Number of principal functions to retrieve

        Returns
        -------
        list of callables
            The principal function up to the k-th
        """
        return basis.BasisFromSmoothFunctions(self.output_smoother.get_functions()[:k], 1, self.domain)

    def get_eig_vals(self, k):
        """
        Parameters
        ----------
        k: int
            Number of eigenvalues to retrieve

        Returns
        -------
        array-like
            The k largest eigenvalues
        """
        return self.eig_vals[:k]

    def get_regressors(self, k):
        return self.output_smoother.get_functions()[:k]

    def predict(self, xlocs):
        return self.output_smoother.predict(xlocs)



class FunctionalPCAFromDiscrete:
    """
    Parameters
    ----------
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain to consider
    n_evals: int
        number of evaluations to use for the discrete approximation
    output_smoother: functional_data.smoothing.Smoother
        The smoother to use to pass from the discrete approximation to functions
    """
    def __init__(self, domain, n_evals, output_smoother):
        self.domain = domain
        self.output_smoother = output_smoother
        self.n_evals = n_evals
        self.eig_vals = None

    def fit(self, Xfuncs):
        """
        Fit the PCA

        Parameters
        ----------
        Xfuncs: iterable of callables
            The functional data
        """
        n = len(Xfuncs)
        space = np.linspace(self.domain[0, 0], self.domain[0, 1], self.n_evals)
        data_mat = np.array([func(space) for func in Xfuncs]).squeeze()
        u, v, w = np.linalg.svd(data_mat, full_matrices=False)
        a = (self.domain[0, 1] - self.domain[0, 0]) / self.n_evals
        self.output_smoother.fit([space for i in range(n)], (1 / np.sqrt(a)) * w)
        self.eig_vals = v ** 2

    def get_principal_basis(self, k):
        """
        Parameters
        ----------
        k: int
            Number of principal functions to retrieve

        Returns
        -------
        list of callables
            The principal function up to the k-th
        """
        return basis.BasisFromSmoothFunctions(self.output_smoother.get_functions()[:k], 1, self.domain)

    def get_eig_vals(self, k):
        """
        Parameters
        ----------
        k: int
            Number of eigenvalues to retrieve

        Returns
        -------
        array-like
            The k largest eigenvalues
        """
        return self.eig_vals[:k]

    def get_regressors(self, k):
        return self.output_smoother.get_functions()[:k]

    def predict(self, xlocs):
        return self.output_smoother.predict(xlocs)
