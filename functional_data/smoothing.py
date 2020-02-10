import numpy as np
from sklearn.linear_model import Ridge, Lasso


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
    w : numpy.ndarray
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
    w : numpy.ndarray
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


class Smoother:

    def __init__(self, basis, regu, method="Ridge"):
        self.basis = basis
        self.regu = regu
        self.method = method
        self.regressors = []

    def fit(self, Xlocs, Xevals):
        n = len(Xlocs)
        self.regressors = []
        for i in range(n):
            if self.method == "Ridge":
                reg = ExpandedRidge(self.regu, self.basis)
            elif self.method == "Lasso":
                reg = ExpandedLasso(self.regu, self.basis)
            else:
                raise ValueError('method should be either "Ridge" or "Lasso"')
            reg.fit(Xlocs[i], Xevals[i])
            self.regressors.append(reg)

    def get_functions(self):
        return self.regressors

    def predict(self, xlocs):
        return np.array([reg.predict(xlocs) for reg in self.regressors])


class LinearInterpSmoother:

    def __init__(self):
        self.regressors = []

    @staticmethod
    def interp_function(xloc, xeval):
        def func_interp(x):
            return np.interp(x.squeeze(), xloc.squeeze(), xeval.squeeze())
        return func_interp

    def fit(self, Xlocs, Xevals):
        self.regressors = [LinearInterpSmoother.interp_function(Xlocs[i], Xevals[i]) for i in range(len(Xlocs))]

    def get_functions(self):
        return self.regressors

    def predict(self, xlocs):
        return np.array([reg(xlocs) for reg in self.regressors])








