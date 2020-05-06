from abc import ABC, abstractmethod
import numpy as np
import functools
from scipy.interpolate import BSpline
from integration import integration
import pywt

from functional_data import smoothing
from functional_data import fpca
from functional_data import functional_algebra as falgebra


def constant_function(domain):
    norm_const = 1 / (domain[0, 1] - domain[0, 0])

    def const_func(z):
        if np.ndim(z) == 0:
            return norm_const
        else:
            return norm_const * np.ones(z.shape[0])
    return const_func


# ######################## Abstract classes ############################################################################

class Basis(ABC):
    """
    Abstract class for set of basis functions

    Parameters
    ----------
    n_basis: int
        Number of basis functions
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    """
    def __init__(self, n_basis, input_dim, domain):
        self.n_basis = n_basis
        self.input_dim = input_dim
        self.gram_matrix = None
        self.domain = np.array(domain)
        if self.domain.ndim == 1:
            self.domain = np.expand_dims(self.domain, axis=0)
        self._gram_mat = None
        super().__init__()

    @abstractmethod
    def compute_matrix(self, X):
        """
        Evaluate the set of basis functions on a given set of values

        Parameters
        ----------
        X : array-like, shape = [n_input, input_dim]
            The input data

        Returns
        -------
        array-like, shape=[n_input, n_basis]
            Matrix of evaluations of the inputs for all basis-function
        """
        pass

    @abstractmethod
    def get_gram_matrix(self):
        pass


class DataDependantBasis(Basis):
    """
    Abstract class for set of basis functions

    Parameters
    ----------
    n_basis: int
        Number of basis functions
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    """
    def __init__(self, n_basis, input_dim, domain):
        super().__init__(n_basis, input_dim, domain)

    @abstractmethod
    def fit(self, Ylocs, Yobs):
        """
        Fit the basis to the data

        Parameters
        ----------
        Ylocs: iterable of array-like
            The input locations, len = n_samples and for the i-th sample, Ylocs[i] has shape = [n_observations_i, 1]
        Yobs: iterable of array-like
            The observations len = n_samples and for the i-th sample, Yobs[i] has shape = [n_observations_i, ]
        """
        pass


# ######################## Classic Bases ###############################################################################

class Basis(ABC):
    """
    Abstract class for set of basis functions

    Parameters
    ----------
    n_basis: int
        Number of basis functions
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    """
    def __init__(self, n_basis, input_dim, domain):
        self.n_basis = n_basis
        self.input_dim = input_dim
        self.domain = np.array(domain)
        if self.domain.ndim == 1:
            self.domain = np.expand_dims(self.domain, axis=0)
        self._gram_mat = None
        super().__init__()

    @abstractmethod
    def compute_matrix(self, X):
        """
        Evaluate the set of basis functions on a given set of values

        Parameters
        ----------
        X : array-like, shape = [n_input, input_dim]
            The input data

        Returns
        -------
        array-like, shape=[n_input, n_basis]
            Matrix of evaluations of the inputs for all basis-function
        """
        pass


class RandomFourierFeatures(Basis):
    """
    Random Fourier Features basis

    Parameters
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    bandwidth: float
        Bandwidth parameter of approximated kernel
    seed: int
        Seed to initialize for each random draw

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    bandwidth: float
        Bandwidth parameter of approximated kernel
    seed: int
        Seed to initialize for each random draw (for reproducible experiments)
    w: array-like
        Random weights
    b: array-like
        Random intercept
    """
    def __init__(self, n_basis, domain, bandwidth, input_dim, seed=0):
        super().__init__(n_basis, input_dim, domain)
        self.bandwidth = bandwidth
        self.input_dim = input_dim
        self.seed = seed
        np.random.seed(self.seed)
        self.w = np.random.normal(0, 1, (self.input_dim, self.n_basis))
        np.random.seed(seed)
        self.b = np.random.uniform(0, 2 * np.pi, (1, self.n_basis))
        self.compute_gram_matrix()
        # self.compute_gram_matrix_bis()

    def compute_gram_matrix(self):
        space = np.linspace(0, 1, 1000)
        mat = self.compute_matrix(space)
        self.gram_mat = (1/1000) * mat.T.dot(mat)
    #
    # def compute_gram_matrix(self):
    #     self.gram_mat = np.zeros((self.n_basis, self.n_basis))
    #     for i in range(self.n_basis):
    #         for j in range(self.n_basis):
    #             if i != j:
    #                 a0 = np.abs(self.bandwidth * (self.w[0, i] - self.w[0, j]))
    #                 a1 = np.abs(self.bandwidth * (self.w[0, i] + self.w[0, j]))
    #                 b0 = self.b[0, i] - self.b[0, j]
    #                 b1 = self.b[0, i] + self.b[0, j]
    #                 entry = 0.5 * ((np.sin(a0 + b0) - np.sin(b0)) / a0 + (np.sin(a1 + b1) - np.sin(b1)) / a1)
    #                 self.gram_mat[i, j] = entry
    #                 self.gram_mat[j, i] = entry
    #             else:
    #                 a = 2 * self.bandwidth * np.abs(self.w[0, i])
    #                 b = 2 * self.b[0, i]
    #                 self.gram_mat[i, i] = 0.5 + 0.5 * (1 / a) * (np.sin(a + b) - np.sin(b))
        # for i in range(self.n_basis - 1):
        #     for j in range(i + 1, self.n_basis):
        #         a0 = np.abs(self.bandwidth * (self.w[0, i] - self.w[0, j]))
        #         a1 = np.abs(self.bandwidth * (self.w[0, i] + self.w[0, j]))
        #         b0 = self.b[0, i] - self.b[0, j]
        #         b1 = self.b[0, i] + self.b[0, j]
        #         entry = 0.5 * ((np.sin(a0 + b0) - np.sin(b0)) / a0 + (np.sin(a1 + b1) - np.sin(b1)) / a1)
        #         self.gram_mat[i, j] = entry
        #         self.gram_mat[j, i] = entry
        # for i in range(self.n_basis):
        #     a = 2 * self.bandwidth * np.abs(self.w[0, i])
        #     b = 2 * self.b[0, i]
        #     self.gram_mat[i, i] = 0.5 + 0.5 * (1 / a) * (np.sin(a + b) - np.sin(b))

    def compute_matrix(self, X):
        n = X.shape[0]
        if X.ndim == 1:
            return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.reshape((n, 1)).dot(self.w) + self.b)
        else:
            return np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * X.dot(self.w) + self.b)

    def add_features(self, n_features):
        """
        Append new random features to the already drawn ones

        Parameters
        ----------
        n_features: int
            Number of features to add

        Returns
        -------
        NoneType
        """
        np.random.seed(self.seed)
        wadd = np.random.normal(0, 1, (self.input_dim, n_features))
        np.random.seed(self.seed)
        badd = np.random.uniform(0, 2 * np.pi, (1, n_features))
        self.w = np.concatenate((self.w, wadd), axis=1)
        self.b = np.concatenate((self.b, badd), axis=1)
        self.n_basis = self.n_basis + n_features

    def get_atom(self, i):
        return lambda x: np.sqrt(2 / self.n_basis) * np.cos(self.bandwidth * self.w[:, i].dot(x) + self.b[:, i])

    def get_gram_matrix(self):
        return self.gram_mat




class BasisFromSmoothFunctions(Basis):
    """
    Basis from list of vectorized functions

    Parameters
    ----------
    funcs: iterable
        Iterable containing callables that can be called on array-like inputs
    input_dim: int
        The input dimension for the functions
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The input dimension for the functions
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    funcs: iterable
        Iterable containing callables that can be called on array-like inputs
    add_constant: bool
        Should the constant function be automatically be added
    """
    def __init__(self, funcs, input_dim, domain, add_constant=False):
        n_basis = len(funcs)
        self.funcs = funcs
        self.add_constant = add_constant
        super().__init__(n_basis + int(add_constant), input_dim, domain)

    def __getitem__(self, item):
        return self.funcs[item]

    def compute_matrix(self, X):
        n = X.shape[0]
        mat = np.zeros((n, self.n_basis))
        for i in range(self.n_basis - int(self.add_constant)):
            mat[:, i] = self.funcs[i](X)
        if self.add_constant:
            mat[:, -1] = 1
        return mat


class FourierBasis(Basis):
    """
    Abstract class for set of basis functions

    Parameters
    ----------
    lower_freq: int
        Minimum frequency to consider
    upper_freq: int
        Maximum frequency to consider
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function

    Attributes
    ----------
    n_basis: int
        Number of basis functions
    input_dim: int
        The number of dimensions of the input space
    domain: array-like, shape = [input_dim, 2]
        Bounds for the domain of the basis function
    freqs: tuple, len = 2
        Frequencies included in the basis
    """
    def __init__(self, lower_freq, upper_freq, domain):
        self.freqs = np.arange(lower_freq, upper_freq)
        if lower_freq == 0:
            n_basis = 2 * (len(self.freqs) - 1) + 1
        else:
            n_basis = 2 * len(self.freqs) + 1
        input_dim = 1
        super().__init__(n_basis, input_dim, domain)

    @staticmethod
    def cos_atom(n, a, b, x):
        return (1 / np.sqrt((b - a) / 2)) * np.cos((2 * np.pi * n * (x - a)) / (b - a))

    @staticmethod
    def sin_atom(n, a, b, x):
        return (1 / np.sqrt((b - a) / 2)) * np.sin((2 * np.pi * n * (x - a)) / (b - a))

    @staticmethod
    def constant_atom(a, b, x):
        return 1 / np.sqrt(b - a) * np.ones(x.shape)

    def compute_matrix(self, X):
        n = X.shape[0]
        mat = np.zeros((n, self.n_basis))
        ncos = self.n_basis // 2
        zero_shift = int(0 in self.freqs)
        a = self.domain[0, 0]
        b = self.domain[0, 1]
        mat[:, 0] = np.array([FourierBasis.constant_atom(a, b, np.squeeze(X))])
        for base in range(zero_shift, ncos):
            mat[:, base] = np.array([FourierBasis.cos_atom(self.freqs[base], a, b, np.squeeze(X))])
            mat[:, ncos + base] = np.array([FourierBasis.sin_atom(self.freqs[base], a, b, np.squeeze(X))])
        return mat

    def get_atoms(self):
        ncos = self.n_basis // 2
        a = self.domain[0, 0]
        b = self.domain[0, 1]
        zero_shift = int(0 in self.freqs)
        cos_atoms = [functools.partial(FourierBasis.constant_atom, a, b)]
        sin_atoms = []
        for base in range(zero_shift, ncos):
            cos_atoms.append(functools.partial(FourierBasis.cos_atom, self.freqs[base], a, b))
            sin_atoms.append(functools.partial(FourierBasis.sin_atom, self.freqs[base], a, b))
        return cos_atoms + sin_atoms

    def get_gram_matrix(self):
        return np.eye(self.n_basis)


class BSplineUniscaleBasis(Basis):
    """
    Parameters
    ----------
    domain: array-like, shape = [1, 2]
        the domain of interest
    n_basis: int
        number of basis to consider (regular repartition into locs_bounds
    locs_bounds: array-like, shape = [1, 2]
        the bounds for the peaks locations of the splines
    width: int
        the width of the splines
    bounds_disc: bool
        should the knots outside of the domain be thresholded to account for possible out of domain discontinuity
    order: int
        the order of the spline, 3 is for cubic spline for instance.
    """
    def __init__(self, domain, n_basis, locs_bounds, width=1.0, bounds_disc=False,
                 order=3, add_constant=True):
        self.locs_bounds = locs_bounds
        self.bounds_disc = bounds_disc
        self.order = order
        self.width = width
        self.knots = BSplineUniscaleBasis.knots_generator(domain, n_basis, locs_bounds, width, bounds_disc, order)
        self.splines = [falgebra.NoNanWrapper(BSpline.basis_element(self.knots[i], extrapolate=False))
                        for i in range(len(self.knots))]
        self.add_constant = add_constant
        self.norms = [np.sqrt(integration.func_scalar_prod(sp, sp, domain)) for sp in self.splines]
        # self.norms = [np.sqrt(integration.func_scalar_prod(sp, sp, domain)[0]) for sp in self.splines]
        input_dim = 1
        super().__init__(n_basis + int(self.add_constant), input_dim, domain)
        self.gram_mat = self.gram_matrix()

    @staticmethod
    def knots_generator(domain, n_basis, locs_bounds, width=1, bounds_disc=False, order=3):
        locs = np.linspace(locs_bounds[0], locs_bounds[1], n_basis, endpoint=True)
        pace = width / (order + 1)
        cardinal_knots = np.arange(-width / 2, width / 2 + pace, pace)
        if not bounds_disc:
            knots = [cardinal_knots + loc for loc in locs]
        else:
            knots = []
            for loc in locs:
                knot = cardinal_knots + loc
                knot[knot < domain[0, 0]] = domain[0, 0]
                knot[knot > domain[0, 1]] = domain[0, 1]
                knots.append(knot)
        return knots

    def gram_matrix(self):
        gram_mat = np.zeros((self.n_basis, self.n_basis))
        funcs = self.splines.copy()
        if self.add_constant:
            funcs.append(constant_function(self.domain))
        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                esti_scalar = integration.func_scalar_prod(funcs[i], funcs[j], self.domain)
                gram_mat[i, j] = (1 / (self.norms[i] * self.norms[j])) * esti_scalar
                gram_mat[j, i] = (1 / (self.norms[i] * self.norms[j])) * esti_scalar
                # gram_mat[i, j] = esti_scalar[0]
                # gram_mat[j, i] = esti_scalar[0]
        return gram_mat

    def get_gram_matrix(self):
        return self.gram_mat

    def compute_matrix(self, X):
        if X.ndim == 1:
            Xreshaped = np.expand_dims(X, axis=1)
        else:
            Xreshaped = X
        evals = [(1 / self.norms[i]) * self.splines[i](Xreshaped) for i in range(len(self.norms))]
        # evals = [sp(Xreshaped) for sp in self.splines]
        constant = np.ones((Xreshaped.shape[0], 1))
        if self.add_constant:
            evals.append(constant)
        evals = np.concatenate(evals, axis=1)
        evals[np.isnan(evals)] = 0
        return evals


class BSplineMultiscaleBasis(Basis):
    """
    Parameters
    ----------
    domain: array-like, shape = [1, 2]
        the domain of interest
    n_basis_1st_scale: int
        number of basis to consider at the initial scale, even repartition in locs_bounds
    n_basis_increase: int
        power of increase of the number of basis at each scale, 2 means for instance double.
    locs_bounds: array-like, shape = [1, 2]
        the bounds for the peaks locations of the splines
    width_init: float
        Width at the initial scale
    dilat: float
        Dilation coefficient, at every scale, the width is divided by this coefficient
    n_dilat: the number of dilations to perform from the initial scale
    bounds_disc: bool
        should the knots outside of the domain be thresholded to account for possible out of domain discontinuity
    order: int
        the order of the spline, 3 is for cubic spline for instance.
    """
    def __init__(self, domain, n_basis_1st_scale, n_basis_increase, locs_bounds, width_init=1.0, dilat=2.0,
                 n_dilat=2, bounds_disc=False, order=3, add_constant=True):
        self.locs_bounds = locs_bounds
        self.bounds_disc = bounds_disc
        self.order = order
        self.add_constant = add_constant
        self.widths = [width_init * (1 / dilat) ** i for i in range(n_dilat)]
        n_basis_per_scale = [int(n_basis_1st_scale * n_basis_increase ** i) for i in range(n_dilat)]
        self.uniscale_bases = [BSplineUniscaleBasis(
            domain, n_basis_per_scale[i], locs_bounds, width=self.widths[i],
            bounds_disc=bounds_disc, order=order, add_constant=False) for i in range(n_dilat)]
        input_dim = 1
        super().__init__(np.sum(np.array(n_basis_per_scale)) + int(self.add_constant), input_dim, domain)

    def compute_matrix(self, X):
        evals = [scale.compute_matrix(X) for scale in self.uniscale_bases]
        constant = np.ones((X.shape[0], 1))
        evals.append(constant)
        return np.concatenate(evals, axis=1)


class UniscaleCompactlySupported(Basis):
    """
    Uniscale basis of compactly supported discrete wavelets

    Parameters
    ----------
    domain: array-like, shape = [1, 2]
        domain of interest
    locs_bounds: array_like, shape = [1, 2]
        bounds for the support of the wavelets considered
    pywt_name: str, {"coif", "db"}
        wavelet name
    moments: int
        number of vanishing moments
    dilat: float
        dilatation coefficient over the undilated mother wavelet
    translat: float
        space between each beginning of wavelet
    approx_level: int
        approx level to consider in pywt
    add_constant: bool
        should the constant function be added
    """
    def __init__(self, domain, locs_bounds, pywt_name="coif", moments=3,
                 dilat=1.0, translat=1.0, approx_level=5, add_constant=True):
        self.pywt_name = pywt_name + str(moments)
        self.dilat = dilat
        self.translat = translat
        phi, psi, x = pywt.Wavelet(self.pywt_name).wavefun(level=approx_level)
        x /= self.dilat
        self.eval_mother = np.sqrt(dilat) * psi
        trans_grid = []
        t = locs_bounds[0, 0]
        while x[-1] + t <= locs_bounds[0, 1]:
            trans_grid.append(t)
            t += translat / dilat
        self.eval_grids = [x + t for t in trans_grid]
        input_dim = 1
        self.add_constant = add_constant
        super().__init__(len(trans_grid) + int(add_constant), input_dim, domain)

    def compute_matrix(self, X):
        n = X.shape[0]
        mat = np.zeros((n, self.n_basis))
        for i in range(self.n_basis - int(self.add_constant)):
            mat[:, i] = np.interp(X.squeeze(), self.eval_grids[i], self.eval_mother)
        if self.add_constant:
            constant = np.ones((X.shape[0], 1))
            return np.concatenate((mat, constant), axis=1)
        else:
            return mat

    def get_gram_matrix(self):
        return np.eye(self.n_basis)


class MultiscaleCompactlySupported(Basis):
    """
    Multiscale basis of compactly supported discrete wavelets

    Parameters
    ----------
    domain: array-like, shape = [1, 2]
        domain of interest
    locs_bounds: array_like, shape = [1, 2]
        bounds for the support of the wavelets considered
    pywt_name: str, {"coif", "db"}
        wavelet name
    moments: int
        number of vanishing moments
    init_dilat: float
        dilatation coefficient over the undilated mother wavelet for the initial scale
    dilat: float
        the dilatation coefficients wy which the scale is divided between each scale
    n_dilat: int
        nmuber of scales
    translat: float
        space between each beginning of wavelet
    approx_level: int
        approx level to consider in pywt
    add_constant: bool
        should the constant function be added
    """
    def __init__(self, domain, locs_bounds, pywt_name="coif", moments=3,
                 init_dilat=1.0, dilat=2.0, n_dilat=2, translat=1.0,
                 approx_level=5, add_constant=True):
        self.scale_bases = []
        self.add_constant = add_constant
        for n in range(n_dilat):
            self.scale_bases.append(UniscaleCompactlySupported(domain, locs_bounds, pywt_name, moments,
                                                               init_dilat * dilat ** n,
                                                               translat, approx_level, add_constant=False))
        n_basis = np.sum([scale.n_basis for scale in self.scale_bases])
        super().__init__(n_basis + int(self.add_constant), 1, domain)

    def compute_matrix(self, X):
        evals = [scale.compute_matrix(X) for scale in self.scale_bases]
        constant = np.ones((X.shape[0], 1))
        if self.add_constant:
            evals.append(constant)
        return np.concatenate(evals, axis=1)

    def get_gram_matrix(self):
        return np.eye(self.n_basis)


# ######################## Data-dependant bases ########################################################################

class FPCABasis(DataDependantBasis):

    def __init__(self, n_basis, input_dim, domain, n_evals, output_smoother=smoothing.LinearInterpSmoother()):
        self.fpca = fpca.FunctionalPCA(domain, n_evals, output_smoother)
        super().__init__(n_basis, input_dim, domain)

    def fit(self, *args):
        self.fpca.fit(*args)

    def compute_matrix(self, X):
        # evals = [self.fpca.predict(X[i])[:self.n_basis] for i in range(len(X))]
        # return np.array(evals)
        n = X.shape[0]
        funcs = self.fpca.get_regressors(self.n_basis)
        mat = np.zeros((n, self.n_basis))
        for i in range(self.n_basis):
            mat[:, i] = funcs[i](X)
        return mat

    def get_gram_matrix(self):
        return np.eye(self.n_basis)


# ######################## Basis generation ############################################################################

SUPPORTED_DICT = {"random_fourier": RandomFourierFeatures,
                  "fourier": FourierBasis,
                  "wavelets": MultiscaleCompactlySupported,
                  "functional_pca": FPCABasis}


def generate_basis(key, kwargs):
    """
    Generate basis from name and keywords arguments

    Parameters
    ----------
    key: {"random_fourier", "fourier", "wavelets", "from_smooth_funcs", "functional_pca"}
        The basis reference name
    kwargs: dict
        key words argument to build the basis in question

    Returns
    -------
    Basis
        Generated basis
    """
    return SUPPORTED_DICT[key](**kwargs)


def set_basis_config(passed_basis):
    """
    Make the difference between what is passed as basis, can be either a Basis instance or a configuration tuple

    Parameters
    ----------
    passed_basis: Basis or tuple
        If tuple is given, must be of the form (basis_key, basis_config) with basis_key in `SUPPORTED_DICT` and
        basis_config a dictionary of basis parameters with the right keys

    Returns
    -------
    tuple,
        a config and a basis, one of them is None
    """
    if isinstance(passed_basis, Basis):
        actual_basis = passed_basis
        config_basis = None
    else:
        config_basis = passed_basis
        actual_basis = None
    return config_basis, actual_basis