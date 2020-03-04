import numpy as np

from model_eval import configs_generation
from functional_regressors import kernels
from functional_regressors import kernel_projection_learning as kproj_learning
from functional_regressors import triple_basis
from functional_regressors import kernel_additive


# ############################### KPL ##################################################################################
def dti_wavs_kpl(ker_sigma, regu, center_output=True, signal_ext=("symmetric", (1, 1)),
                 decrease_base=1, pywt_name="db", moments=2, init_dilat=1.0, translat=1.0, dilat=2,
                 approx_level=5, add_constant=True, domain=np.array([[0, 1]]), locs_bounds=np.array([[0, 1]])):
    # Wavelets output basses
    output_basis_params = {"pywt_name": pywt_name, "moments": moments, "init_dilat": init_dilat, "translat": translat,
                           "dilat": dilat, "approx_level": approx_level, "add_constant": add_constant,
                           "locs_bounds": locs_bounds, "domain": domain}
    output_bases = configs_generation.subconfigs_combinations("wavelets", output_basis_params,
                                                              exclude_list=["domain", "locs_bounds"])
    # Gaussian kernel
    ker = kernels.GaussianScalarKernel(ker_sigma, normalize=False)
    # Penalize power
    output_matrix_params = {"decrease_base": decrease_base}
    output_matrices = configs_generation.subconfigs_combinations("wavelets_pow", output_matrix_params)
    # Generate full configs
    params = {"kernel_scalar": ker, "B": output_matrices, "output_basis": output_bases,
              "regu": regu, "center_output": center_output, "signal_ext": signal_ext}
    configs = configs_generation.configs_combinations(params, exclude_list=["signal_ext"])
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def speech_fpca_penpow_kpl(ker_sigma, regu, n_fpca, n_evals_fpca, decrease_base, domain=np.array([[0, 1]])):
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
              "regu": regu, "center_output": True}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


# ############################### KAM ##################################################################################

def dti_kam(kx_sigma, ky_sigma, keval_sigma, regu, n_fpca, n_evals_fpca, n_evals_in,
            n_evals_out, domain_in=np.array([[0, 1]]), domain_out=np.array([[0, 1]])):
    kx = kernels.GaussianScalarKernel(kx_sigma, normalize=False)
    ky = kernels.GaussianScalarKernel(ky_sigma, normalize=False)
    keval = kernels.GaussianScalarKernel(keval_sigma, normalize=False)
    params = {"regu": regu, "kerlocs_in": kx, "kerlocs_out": ky, "kerevals": keval,
              "n_fpca": n_fpca, "n_evals_fpca": n_evals_fpca,
              "n_evals_in": n_evals_in, "n_evals_out": n_evals_out,
              "domain_in": domain_in, "domain_out": domain_out}
    configs = configs_generation.configs_combinations(params, exclude_list=("domain_in", "domain_out"))
    # Create list of regressors from that config
    regs = [kernel_additive.KernelAdditiveModel(**config) for config in configs]
    return configs, regs


# ############################### 3BE ##################################################################################

def dti_3be_fourier(ker_sigma, regu, center_output, max_freq_in, max_freq_out,
                    n_rffs, rffs_seed, domain_in, domain_out, signal_ext_input=None, signal_ext_output=None):
    input_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_in, "domain": domain_in}
    output_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_out, "domain": domain_out}
    rffs_basis_dict = {"n_basis": n_rffs, "domain": domain_out, "bandwidth": ker_sigma, "seed": rffs_seed}
    rffs_basis = ("random_fourier", rffs_basis_dict)
    bases_in = configs_generation.subconfigs_combinations("fourier", input_basis_dict, exclude_list=["domain"])
    bases_out = configs_generation.subconfigs_combinations("fourier", output_basis_dict, exclude_list=["domain"])
    # Generate full configs
    params = {"basis_in": bases_in, "basis_out": bases_out, 'basis_rffs': rffs_basis, "regu": regu,
              "center_output": center_output, "signal_ext_input": signal_ext_input,
              "signal_ext_output": signal_ext_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["signal_ext_input",
                                                                            "signal_ext_output",
                                                                            'basis_rffs'])
    # Create list of regressors from that config
    regs = [triple_basis.TripleBasisEstimator(**config) for config in configs]
    return configs, regs
