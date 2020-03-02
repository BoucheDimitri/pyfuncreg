import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, Lasso


# ##################### Expanded regressions ###########################################################################

class ExpandedRidge:
    """
    Ridge regression with basis expansion

    Parameters
    ----------
    basis : functional_data.basis.Basis
        set of basis functions
    lamb : float
        regularization parameter

    Attributes
    ----------
    basis : functional_data.basis.Basis
        set of basis functions
    lamb : float
        regularization parameter
    w : array_like
        regression weights
    """
    def __init__(self, lamb, basis):
        self.lamb = lamb
        self.basis = basis
        self.w = None

    def fit(self, X, y):
        Z = self.basis.compute_matrix(X)
        ridge = Ridge(alpha=self.lamb, fit_intercept=False)
        ridge.fit(Z, y)
        self.w = ridge.coef_.flatten()

    def predict(self, X):
        Z = self.basis.compute_matrix(X)
        return Z.dot(self.w)

    def __call__(self, X):
        return self.predict(X)


class ExpandedLasso:
    """
    Ridge regression with basis expansion

    Parameters
    ----------
    basis : functional_data.basis.Basis
        set of basis functions
    lamb : float
        regularization parameter

    Attributes
    ----------
    basis : functional_data.basis.Basis
        set of basis functions
    lamb : float
        regularization parameter
    w : array_like
        regression weights
    """
    def __init__(self, lamb, basis):
        self.lamb = lamb
        self.basis = basis
        self.w = None

    def fit(self, X, y):
        Z = self.basis.compute_matrix(X)
        lasso = Lasso(alpha=self.lamb, fit_intercept=False, max_iter=5000)
        lasso.fit(Z, y)
        self.w = lasso.coef_.flatten()

    def predict(self, X):
        Z = self.basis.compute_matrix(X)
        return Z.dot(self.w)

    def __call__(self, X):
        return self.predict(X)


# ##################### Smoothers ######################################################################################

class Smoother(ABC):

    def __init__(self):
        """
        Perform smoothing on a whole sample of discretized functions
        """
        self.regressors = []
        super().__init__()

    @abstractmethod
    def fit(self, Xlocs, Xobs):
        """
        Fit the smoother

        Parameters
        ----------
        Xlocs : iterable of array_like
            The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
        Xobs : iterable of array_like
            The observations len = n_samples and for the i-th sample, Yobs[i] has shape = [n_observations_i, ]
        """
        pass

    @abstractmethod
    def get_functions(self):
        """
        Getter for the regressors attribute

        Returns
        -------
        iterable of callable
            The list of smoothed functions
        """
        pass

    @abstractmethod
    def predict(self, locs):
        """
        Evaluate at new locations using the smoothed functions

        Parameters
        ----------
        locs : array_like
            The locations where to evaluate

        Returns
        -------
        array_like
            Array of evaluations
        """
        pass


class RegressionSmoother(Smoother):
    """
    Parameters
    ----------
    basis: functional_data.basis.Basis
        The basis on which to perform the smoothing
    regu: float
        The regularization for the smoothing
    method: str, {"Ridge", "Lasso"}
        Whether to use Ridge or Lasso.
    """
    def __init__(self, basis, regu, method="Ridge"):
        self.basis = basis
        self.regu = regu
        self.method = method
        super().__init__()

    def fit(self, Xlocs, Xobs):
        n = len(Xlocs)
        self.regressors = []
        for i in range(n):
            if self.method == "Ridge":
                reg = ExpandedRidge(self.regu, self.basis)
            elif self.method == "Lasso":
                reg = ExpandedLasso(self.regu, self.basis)
            else:
                raise ValueError('method should be either "Ridge" or "Lasso"')
            reg.fit(Xlocs[i], Xobs[i])
            self.regressors.append(reg)

    def get_functions(self):
        return self.regressors

    def predict(self, locs):
        return np.array([reg.predict(locs) for reg in self.regressors])


# TODO: Beware, extrapolation is performed using the last value, not linearily

class LinearInterpSmoother(Smoother):
    """
    RegressionSmoother using linear interpolation
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def interp_function(xloc, xeval):
        def func_interp(x):
            return np.interp(x.squeeze(), xloc.squeeze(), xeval.squeeze())
        return func_interp

    def fit(self, Xlocs, Xobs):
        self.regressors = [LinearInterpSmoother.interp_function(Xlocs[i], Xobs[i]) for i in range(len(Xlocs))]

    def get_functions(self):
        return self.regressors

    def predict(self, locs):
        return np.array([reg(locs) for reg in self.regressors])








