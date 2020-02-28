from abc import ABC, abstractmethod
import numpy as np


class OutputMatrix(ABC):
    """
    Abstract class for output matrix
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_matrix(self, output_basis):
        """

        Parameters
        ----------
        output_basis: functional_data.basis.Basis
            The output basis

        Returns
        -------
        array-like
            The output matrix
        """
        pass


class Eye(OutputMatrix):

    def __init__(self):
        super().__init__()

    def get_matrix(self, output_basis):
        return np.eye(output_basis.n_basis)


class Pow(OutputMatrix):

    def __init__(self, decrease_base):
        """
        Power penalization with basis index

        Parameters
        ----------
        decrease_base: float
            Penalization is done in 1 / decrease_base^j
        """
        self.decrease_base = decrease_base
        super().__init__()

    def get_matrix(self, output_basis):
        """
        Parameters
        ----------
        output_basis: functional_data.basis.MultiscaleCompactlySupported
            The wavelet dictionary

        Returns
        -------
        array-like, shape = [output_basis.n_basis, output_basis.n_basis]
            The output matrix
        """
        freqs_penalization = np.array([(1 / self.decrease_base) ** j for j in range(output_basis.n_basis)])
        return np.diag(freqs_penalization)


class WaveletsPow(OutputMatrix):

    def __init__(self, decrease_base):
        """
        Power penalization with scale

        Parameters
        ----------
        decrease_base: float
            Penalization is done in 1 / decrease_base^j
        """
        self.decrease_base = decrease_base
        super().__init__()

    def get_matrix(self, output_basis):
        """
        Parameters
        ----------
        output_basis: functional_data.basis.MultiscaleCompactlySupported
            The wavelet dictionary

        Returns
        -------
        array-like, shape = [output_basis.n_basis, output_basis.n_basis]
            The output matrix
        """
        n_basis_scales = [b.n_basis for b in output_basis.scale_bases]
        freqs_penalization = []
        for j in range(len(n_basis_scales)):
            freqs_penalization += [(1 / self.decrease_base) ** j for i in range(n_basis_scales[j])]
        if output_basis.add_constant:
            freqs_penalization += [1]
        freqs_penalization = np.array(freqs_penalization)
        return np.diag(freqs_penalization)


class WaveletsLinear(OutputMatrix):

    def __init__(self):
        """
        Linear penalization with scale for wavelet bases
        """
        super().__init__()

    def get_matrix(self, output_basis):
        """
        Parameters
        ----------
        output_basis: functional_data.basis.MultiscaleCompactlySupported
            The wavelet dictionary

        Returns
        -------
        array-like, shape = [output_basis.n_basis, output_basis.n_basis]
            The output matrix
        """
        n_basis_scales = [b.n_basis for b in output_basis.scale_bases]
        freqs_penalization = []
        for j in range(len(n_basis_scales)):
            freqs_penalization += [1 / (j + 1) for i in range(n_basis_scales[j])]
        if output_basis.add_constant:
            freqs_penalization += [1]
        freqs_penalization = np.array(freqs_penalization)
        return np.diag(freqs_penalization)


# ########################### Generate #################################################################################

SUPPORTED_DICT = {"pow": Pow, "wavelets_pow": WaveletsPow, "wavelets_linear": WaveletsLinear, "eye": Eye}


def generate_output_matrix(key, kwargs):
    """
    Generate output matrix of type key from kwargs

    Parameters
    ----------
    key: {"wavelets_pow", "wavelets_linear"}
        The output matrix reference name
    kwargs: dict
        keywords argument to build the basis in question

    Returns
    -------
    OutputMatrix
        Generated output matrix
    """
    return SUPPORTED_DICT[key](**kwargs)