from abc import ABC
import numpy as np


class ScalarKernel(ABC):
    """
    Abstract class for scalar valued kernels
    """

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def __call__(self, x0, x1):
        """
        Compute the kernel or the kernel matrix between two vectors or set of vectors

        Parameters
        ----------
        x0 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]
        x1 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]

        Returns
        -------
        float or numpy.ndarray
            For a single evaluation returns a float, else returns a matrix with shape = [n_samples_x1, n_samples_x0]
        """
        pass


class GaussianScalarKernel(ScalarKernel):
    """
    Gaussian Kernel:

    .. math::
        k(x_0, x_1) = \\exp \\left ( \\frac { \\|x_1 - x_0 \\|^2 }{ \\sigma^2} \\right )

    Parameters
    ----------
    stdev : float
        the standard deviation parameter

    Attributes
    ----------
    stdev : float
        the standard deviation parameter
    """

    def __init__(self, stdev, normalize, normalize_dist=False):
        super().__init__(normalize=normalize)
        self.normalize_dist = normalize_dist
        self.stdev = stdev

    def __call__(self, x0, x1):
        """
        Compute the kernel or the kernel matrix between two vectors or set of vectors

        Parameters
        ----------
        x0 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]
        x1 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]

        Returns
        -------
        float or numpy.ndarray
            For a single evaluation returns a float, else returns a matrix with shape = [n_samples_x1, n_samples_x0]
        """
        if isinstance(x0, list) or isinstance(x0, tuple):
            x0_reshaped = np.array(x0)
        else:
            x0_reshaped = x0
        if x0_reshaped.ndim == 1:
            x0_reshaped = np.expand_dims(x0_reshaped, axis=0)
        if isinstance(x1, list) or isinstance(x1, tuple):
            x1_reshaped = np.array(x1)
        else:
            x1_reshaped = x1
        if x1_reshaped.ndim == 1:
            x1_reshaped = np.expand_dims(x1_reshaped, axis=0)
        n = x0_reshaped.shape[0]
        m = x1_reshaped.shape[0]
        K = np.zeros((m, n))
        for j in range(m):
            K[j, :] = np.power(np.linalg.norm(x0_reshaped - x1_reshaped[j], axis=1), 2)
            # K[j, :] = np.exp(- np.power(np.linalg.norm(x0_reshaped - x1_reshaped[j], axis=1), 2) / self.stdev ** 2)
        if self.normalize_dist:
            K *= (1 / K.mean())
        if n == 1 and m == 1:
            return K[0, 0]
        else:
            return np.exp(- K / self.stdev ** 2)


class LinearScalarKernel(ScalarKernel):

    def __init__(self, normalize):
        super().__init__(normalize=normalize)

    def __call__(self, x0, x1):
        if isinstance(x0, list):
            x0_reshaped = np.array(x0).squeeze()
        else:
            x0_reshaped = x0
        if isinstance(x1, list):
            x1_reshaped = np.array(x1).squeeze()
        else:
            x1_reshaped = x1
        return x1_reshaped.dot(x0_reshaped.T)


class SumOfScalarKernel(ScalarKernel):

    def __init__(self, kernels_list, normalize):
        self.kernels_list = kernels_list
        super().__init__(normalize=normalize)

    def __call__(self, x0, x1):
        nkers = len(self.kernels_list)
        n = len(x0)
        m = len(x1)
        X0_split = [[x0[i][:, j] for i in range(n)] for j in range(nkers)]
        X1_split = [[x1[i][:, j] for i in range(m)] for j in range(nkers)]
        Ks = np.array([self.kernels_list[k](X0_split[k], X1_split[k]) for k in range(nkers)])
        # return Ks
        return np.sum(Ks, axis=0)


class LaplaceScalarKernel(ScalarKernel):
    """
    Laplace kernel:

    .. math::
        k(x_0, x_1) = \\exp \\left ( \\frac { \\|x_1 - x_0 \\|}{ \\sigma^2} \\right )

    Parameters
    ----------
    stdev : float
        the standard deviation parameter

    Attributes
    ----------
    stdev : float
        the standard deviation parameter
    """

    def __init__(self, band, normalize):
        super().__init__(normalize=normalize)
        self.band = band

    def __call__(self, x0, x1):
        """
        Compute the kernel or the kernel matrix between two vectors or set of vectors

        Parameters
        ----------
        x0 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]
        x1 : array_like
            Must be either of shape x0.shape = [n_features] or x0.shape = [n_samples, n_features]

        Returns
        -------
        float or array_like
            For a single evaluation returns a float, else returns a matrix with shape = [n_samples_x1, n_samples_x0]
        """
        if isinstance(x0, list) or isinstance(x0, tuple):
            x0_reshaped = np.array(x0)
        else:
            x0_reshaped = x0
        if x0_reshaped.ndim == 1:
            x0_reshaped = np.expand_dims(x0_reshaped, axis=0)
        if isinstance(x1, list) or isinstance(x1, tuple):
            x1_reshaped = np.array(x1)
        else:
            x1_reshaped = x1
        if x1_reshaped.ndim == 1:
            x1_reshaped = np.expand_dims(x1_reshaped, axis=0)
        n = x0_reshaped.shape[0]
        m = x1_reshaped.shape[0]
        K = np.zeros((m, n))
        for j in range(m):
            K[j, :] = np.exp(- np.linalg.norm(x0_reshaped - x1_reshaped[j], axis=1, ord=1) / self.band)
        if n == 1 and m == 1:
            return K[0, 0]
        else:
            return K
