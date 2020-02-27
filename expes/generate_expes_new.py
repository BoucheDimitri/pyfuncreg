import numpy as np

from model_eval import configs_generation
from functional_regressors import kernels
from functional_regressors import kernel_projection_learning as kproj_learning
from functional_regressors import regularization

# TODO finish this
def speech_fpca_gaussker_kpl(domain, ker_sigma, n_fpca, n_evals_fpca):
    # FPCA output basis
    output_basis_params = {"n_basis": n_fpca, "input_dim": 1, "domain": domain, "n_evals": n_evals_fpca}
    output_bases = configs_generation.subconfigs_combinations("functional_pca",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    ker_sigmas = np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Penalize power

