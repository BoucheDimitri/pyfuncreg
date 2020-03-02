import numpy as np

from model_eval import configs_generation
from functional_regressors import kernels
from functional_regressors import kernel_projection_learning as kproj_learning


# ############################### KPL ##################################################################################
def dti_wavs_kpl(ker_sigma, regus, center_output=True, signal_ext=("symmetric", (1, 1)),
                 decrease_base=1, pywt_name="db", moments=2, init_dilat=1.0, translat=1.0,
                 approx_level=5, add_constant=True, domain=np.array([[0, 1]]), locs_bounds=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"pywt_name": pywt_name, "moments": moments, "init_dilat": init_dilat, "translat": translat,
                           "approx_level": approx_level, "add_constant": add_constant,
                           "locs_bounds": locs_bounds, "domain": domain}
    output_bases = configs_generation.subconfigs_combinations("wavelets", output_basis_params,
                                                              exclude_list=["domain", "locs_bounds"])
    # Sum of Gaussian kernels
    ker = kernels.GaussianScalarKernel(ker_sigma, normalize=False)
    # Penalize power
    output_matrix_params = {"decrease_base": decrease_base}
    output_matrices = configs_generation.subconfigs_combinations("wavelets_pow", output_matrix_params)
    # Generate full configs
    params = {"kernel_scalar": ker, "B": output_matrices, "output_basis": output_bases,
              "regu": regus, "center_output": center_output, "signal_ext": signal_ext}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def speech_fpca_penpow_kpl(ker_sigma, regus, n_fpca, n_evals_fpca, decrease_base, domain=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"n_basis": n_fpca, "input_dim": 1, "domain": domain, "n_evals": n_evals_fpca}
    output_bases = configs_generation.subconfigs_combinations("functional_pca",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    ker_sigmas = ker_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Penalize power
    output_matrix_params = {"decrease_base": decrease_base}
    output_matrices = configs_generation.subconfigs_combinations("pow", output_matrix_params)
    # Generate full configs
    params = {"kernel_scalar": multi_ker, "B": output_matrices, "output_basis": output_bases,
              "regu": regus, "center_output": True}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


