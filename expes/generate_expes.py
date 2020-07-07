import numpy as np
from collections.abc import Iterable

from model_eval import configs_generation
from data import toy_data_spline
from functional_data import basis
from functional_regressors import kernels, kernel_projection_learning as kproj_learning, \
    triple_basis, kernel_additive, ovkernel_ridge, kernel_estimator


# ############################### KPL ##################################################################################
def toy_spline_kpl_corr(kernel_sigma, regu, tasks_correl):
    # Spline dict
    locs_bounds = np.array([toy_data_spline.BOUNDS_FREQS[0], toy_data_spline.BOUNDS_FREQS[1]])
    domain = toy_data_spline.DOM_OUTPUT
    func_dict = basis.BSplineUniscaleBasis(domain, toy_data_spline.BOUNDS_FREQS[-1],
                                           locs_bounds, width=toy_data_spline.WIDTH, add_constant=False)
    # Scalar kernel
    gauss_ker = kernels.GaussianScalarKernel(kernel_sigma, normalize=False)
    # Operator valued kernel matrix
    output_matrix_params = {"omega": tasks_correl, "dim": func_dict.n_basis}
    # output_matrices = configs_generation.subconfigs_combinations("chain_graph", output_matrix_params)
    output_matrices = configs_generation.subconfigs_combinations("neighbors_correl", output_matrix_params)
    params = {"kernel": gauss_ker, "regu": regu,  "B": output_matrices, "basis_out": func_dict, "center_output": False}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def toy_spline_kpl_corr2(kernel_sigma, regu, tasks_correl):
    # Spline dict
    locs_bounds = np.array([toy_data_spline.FREQS[0], toy_data_spline.FREQS[-1]])
    freqs_bounds = toy_data_spline.FREQS[0], toy_data_spline.FREQS[-1]
    width = toy_data_spline.WIDTH
    domain = np.expand_dims(np.array([freqs_bounds[0] - width/2, freqs_bounds[1] + width/2]), axis=0)
    func_dict = basis.BSplineUniscaleBasis(domain, freqs_bounds[1], locs_bounds, width=width, add_constant=False)
    # Scalar kernel
    gauss_ker = kernels.GaussianScalarKernel(kernel_sigma, normalize=False)
    # Operator valued kernel matrix
    output_matrix_params = {"omega": tasks_correl}
    # output_matrices = configs_generation.subconfigs_combinations("chain_graph", output_matrix_params)
    output_matrices = configs_generation.subconfigs_combinations("all_related", output_matrix_params)
    params = {"kernel": gauss_ker, "regu": regu,  "B": output_matrices, "basis_out": func_dict, "center_output": False}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def toy_spline_kpl2(kernel_sigma, regu):
    # Spline dict
    locs_bounds = np.array([toy_data_spline.FREQS[0], toy_data_spline.FREQS[-1]])
    freqs_bounds = toy_data_spline.FREQS[0], toy_data_spline.FREQS[-1]
    width = toy_data_spline.WIDTH
    domain = np.expand_dims(np.array([freqs_bounds[0] - width/2, freqs_bounds[1] + width/2]), axis=0)
    func_dict = basis.BSplineUniscaleBasis(domain, freqs_bounds[1], locs_bounds, width=width, add_constant=False)
    # Scalar kernel
    gauss_ker = kernels.GaussianScalarKernel(kernel_sigma, normalize=False)
    # Operator valued kernel matrix
    B = np.eye(func_dict.n_basis)
    regs = [kproj_learning.SeperableKPL(r, gauss_ker, B, func_dict, center_output=False) for r in regu]
    configs = configs_generation.configs_combinations({"regu": regu})
    return configs, regs


def toy_spline_kpl(kernel_sigma, regu):
    # Spline dict
    locs_bounds = np.array([toy_data_spline.BOUNDS_FREQS[0], toy_data_spline.BOUNDS_FREQS[1]])
    domain = toy_data_spline.DOM_OUTPUT
    func_dict = basis.BSplineUniscaleBasis(domain, toy_data_spline.BOUNDS_FREQS[-1],
                                           locs_bounds, width=toy_data_spline.WIDTH, add_constant=False)
    # Scalar kernel
    gauss_ker = kernels.GaussianScalarKernel(kernel_sigma, normalize=False)
    # Operator valued kernel matrix
    B = np.eye(func_dict.n_basis)
    regs = [kproj_learning.SeperableKPL(r, gauss_ker, B, func_dict, center_output=False) for r in regu]
    configs = configs_generation.configs_combinations({"regu": regu})
    return configs, regs


def dti_wavs_kpl(kernel_sigma, regu, center_output=True, decrease_base=1, pywt_name="db", moments=2, n_dilat=4,
                 init_dilat=1.0, translat=1.0, dilat=2, approx_level=5, add_constant=True,
                 domain=np.array([[0, 1]]), locs_bounds=np.array([[0, 1]])):
    # Wavelets output basses
    output_basis_params = {"pywt_name": pywt_name, "moments": moments, "init_dilat": init_dilat, "translat": translat,
                           "dilat": dilat, "n_dilat": n_dilat, "approx_level": approx_level,
                           "add_constant": add_constant, "locs_bounds": locs_bounds, "domain": domain}
    output_bases = configs_generation.subconfigs_combinations("wavelets", output_basis_params,
                                                              exclude_list=["domain", "locs_bounds"])
    # Gaussian kernel
    kernel = kernels.GaussianScalarKernel(kernel_sigma, normalize=False)
    # Penalize power
    output_matrix_params = {"decrease_base": decrease_base}
    output_matrices = configs_generation.subconfigs_combinations("wavelets_pow", output_matrix_params)
    # Generate full configs
    params = {"kernel": kernel, "B": output_matrices, "basis_out": output_bases,
              "regu": regu, "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def speech_fpca_penpow_kpl(kernel_sigma, regu, n_fpca, n_evals_fpca, decrease_base, domain=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"n_basis": n_fpca, "input_dim": 1, "domain": domain, "n_evals": n_evals_fpca}
    output_bases = configs_generation.subconfigs_combinations("functional_pca",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    kernel_sigmas = kernel_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kernel_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Penalize power
    output_matrix_params = {"decrease_base": decrease_base}
    output_matrices = configs_generation.subconfigs_combinations("pow", output_matrix_params)
    # Generate full configs
    params = {"kernel": multi_ker, "B": output_matrices, "basis_out": output_bases,
              "regu": regu, "center_output": True}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def speech_fourier_kpl(kernel_sigma, regu, n_freqs, center_output, domain=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"lower_freq": 0, "upper_freq": n_freqs, "domain": domain}
    output_bases = configs_generation.subconfigs_combinations("fourier",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    kernel_sigmas = kernel_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kernel_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Penalize power
    output_matrix_params = {}
    output_matrices = configs_generation.subconfigs_combinations("eye", output_matrix_params)
    # Generate full configs
    params = {"kernel": multi_ker, "B": output_matrices, "basis_out": output_bases,
              "regu": regu, "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


def speech_rffs_kpl(kernel_sigma, regu, n_rffs, rffs_sigma, seed_rffs, center_output, domain=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"n_basis": n_rffs, "bandwidth": rffs_sigma, "input_dim": 1, "domain": domain, "seed": seed_rffs}
    output_bases = configs_generation.subconfigs_combinations("random_fourier",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    kernel_sigmas = kernel_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kernel_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Penalize power
    output_matrix_params = {}
    output_matrices = configs_generation.subconfigs_combinations("eye", output_matrix_params)
    # Generate full configs
    params = {"kernel": multi_ker, "B": output_matrices, "basis_out": output_bases,
              "regu": regu, "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [kproj_learning.SeperableKPL(**config) for config in configs]
    return configs, regs


# ############################### KAM ##################################################################################

def kernels_generator_kam(kin_sigma, kout_sigma, keval_sigma):
    if isinstance(kin_sigma, Iterable):
        kernels_in = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in kin_sigma]
    else:
        kernels_in = kernels.GaussianScalarKernel(kin_sigma, normalize=False)
    if isinstance(kout_sigma, Iterable):
        kernels_out = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in kout_sigma]
    else:
        kernels_out = kernels.GaussianScalarKernel(kout_sigma, normalize=False)
    if isinstance(keval_sigma, Iterable):
        kernels_eval = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in keval_sigma]
    else:
        kernels_eval = kernels.GaussianScalarKernel(keval_sigma, normalize=False)
    return kernels_in, kernels_out, kernels_eval


def dti_kam(kin_sigma, kout_sigma, keval_sigma, regu, n_fpca, n_evals_fpca, n_evals_in,
            n_evals_out, domain_in=np.array([[0, 1]]), domain_out=np.array([[0, 1]])):
    kernels_in, kernels_out, kernels_eval = kernels_generator_kam(kin_sigma, kout_sigma, keval_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out, "kernel_eval": kernels_eval,
              "n_fpca": n_fpca, "n_evals_fpca": n_evals_fpca,
              "n_evals_in": n_evals_in, "n_evals_out": n_evals_out,
              "domain_in": domain_in, "domain_out": domain_out}
    configs = configs_generation.configs_combinations(params, exclude_list=("domain_in", "domain_out"))
    # Create list of regressors from that config
    regs = [kernel_additive.KernelAdditiveModelBis(**config) for config in configs]
    return configs, regs


# ############################### 3BE ##################################################################################
def toy_2be(ker_sigma, regu, domain=np.array([[0, 1]])):
    locs_bounds = np.array([toy_data_spline.BOUNDS_FREQS[0], toy_data_spline.BOUNDS_FREQS[1]])
    basis_out = basis.FPCAOrthoSplines(domain, toy_data_spline.BOUNDS_FREQS[-1],
                                       locs_bounds, width=toy_data_spline.WIDTH, add_constant=False)
    gauss_ker = kernels.GaussianScalarKernel(ker_sigma, normalize=False)
    # Generate full configs
    params = {"kernel": gauss_ker, "basis_out": basis_out, "regu": regu, "center_output": False}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.BiBasisEstimator(**config) for config in configs]
    return configs, regs

def toy_2be_four(ker_sigma, regu, max_freq_out, domain=np.array([[0, 1]])):
    output_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_out, "domain": domain}
    bases_out = configs_generation.subconfigs_combinations("fourier", output_basis_dict, exclude_list=["domain"])
    gauss_ker = kernels.GaussianScalarKernel(ker_sigma, normalize=False)
    # Generate full configs
    params = {"kernel": gauss_ker, "basis_out": bases_out, "regu": regu, "center_output": False}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.BiBasisEstimator(**config) for config in configs]
    return configs, regs

def toy_3be_fpcasplines(ker_sigma, regu, center_output, max_freq_in, n_rffs, rffs_seed):
    input_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_in, "domain": toy_data_spline.DOM_INPUT}
    locs_bounds = np.array([toy_data_spline.BOUNDS_FREQS[0], toy_data_spline.BOUNDS_FREQS[1]])
    domain = toy_data_spline.DOM_OUTPUT
    basis_out = basis.FPCAOrthoSplines(domain, toy_data_spline.BOUNDS_FREQS[-1],
                                       locs_bounds, width=toy_data_spline.WIDTH, add_constant=False)
    rffs_basis_dict = {"n_basis": n_rffs, "domain": domain, "bandwidth": ker_sigma, "seed": rffs_seed}
    bases_in = configs_generation.subconfigs_combinations("fourier", input_basis_dict, exclude_list=["domain"])
    bases_rffs = configs_generation.subconfigs_combinations("random_fourier",
                                                            rffs_basis_dict, exclude_list=["domain"])
    # Generate full configs
    params = {"basis_in": bases_in, "basis_out": basis_out, 'basis_rffs': bases_rffs, "regu": regu,
              "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.TripleBasisEstimator(**config) for config in configs]
    return configs, regs


def dti_3be_fourier(ker_sigma, regu, center_output, max_freq_in, max_freq_out,
                    n_rffs, rffs_seed, domain_in, domain_out):
    input_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_in, "domain": domain_in}
    output_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_out, "domain": domain_out}
    rffs_basis_dict = {"n_basis": n_rffs, "domain": domain_out, "bandwidth": ker_sigma, "seed": rffs_seed}
    bases_in = configs_generation.subconfigs_combinations("fourier", input_basis_dict, exclude_list=["domain"])
    bases_out = configs_generation.subconfigs_combinations("fourier", output_basis_dict, exclude_list=["domain"])
    bases_rffs = configs_generation.subconfigs_combinations("random_fourier",
                                                            rffs_basis_dict, exclude_list=["domain"])
    # Generate full configs
    params = {"basis_in": bases_in, "basis_out": bases_out, 'basis_rffs': bases_rffs, "regu": regu,
              "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.TripleBasisEstimator(**config) for config in configs]
    return configs, regs


def dti_3be_wavs(kernel_sigma, regu, center_output, n_rffs, rffs_seed,
                 pywt_name_in="db", moments_in=2, n_dilat_in=4,
                 init_dilat_in=1.0, translat_in=1.0, dilat_in=2, approx_level_in=6, add_constant_in=True,
                 domain_in=np.array([[0, 1]]), locs_bounds_in=np.array([[0, 1]]),
                 pywt_name_out="db", moments_out=2, n_dilat_out=4, init_dilat_out=1.0, translat_out=1.0,
                 dilat_out=2, approx_level_out=6, add_constant_out=True,
                 domain_out=np.array([[0, 1]]), locs_bounds_out=np.array([[0, 1]])):
    rffs_basis_dict = {"n_basis": n_rffs, "domain": domain_out, "bandwidth": kernel_sigma, "seed": rffs_seed}
    bases_rffs = configs_generation.subconfigs_combinations("random_fourier",
                                                            rffs_basis_dict, exclude_list=["domain"])
    # Wavelets output basses
    input_basis_params = {"pywt_name": pywt_name_in, "moments": moments_in, "init_dilat": init_dilat_in,
                          "translat": translat_in, "dilat": dilat_in, "n_dilat": n_dilat_in,
                          "approx_level": approx_level_in, "add_constant": add_constant_in,
                          "locs_bounds": locs_bounds_in, "domain": domain_in}
    output_basis_params = {"pywt_name": pywt_name_out, "moments": moments_out, "init_dilat": init_dilat_out,
                           "translat": translat_out, "dilat": dilat_out, "n_dilat": n_dilat_out,
                           "approx_level": approx_level_out, "add_constant": add_constant_out,
                           "locs_bounds": locs_bounds_out, "domain": domain_out}
    bases_in = configs_generation.subconfigs_combinations("wavelets", input_basis_params,
                                                          exclude_list=["domain", "locs_bounds"])
    bases_out = configs_generation.subconfigs_combinations("wavelets", output_basis_params,
                                                           exclude_list=["domain", "locs_bounds"])
    # Generate full configs
    params = {"basis_in": bases_in, "basis_out": bases_out, 'basis_rffs': bases_rffs, "regu": regu,
              "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.TripleBasisEstimator(**config) for config in configs]
    return configs, regs


def dti_3be_fourwavs(kernel_sigma, regu, center_output, max_freq_in, n_rffs, rffs_seed,
                     pywt_name="db", moments=2, n_dilat=4, init_dilat=1.0, translat=1.0,
                     dilat=2, approx_level=5, add_constant=True, domain_in=np.array([[0, 1]]),
                     domain_out=np.array([[0, 1]]), locs_bounds_out=np.array([[0, 1]])):
    input_basis_dict = {"lower_freq": 0, "upper_freq": max_freq_in, "domain": domain_in}
    rffs_basis_dict = {"n_basis": n_rffs, "domain": domain_out, "bandwidth": kernel_sigma, "seed": rffs_seed}
    bases_in = configs_generation.subconfigs_combinations("fourier", input_basis_dict, exclude_list=["domain"])
    bases_rffs = configs_generation.subconfigs_combinations("random_fourier",
                                                            rffs_basis_dict, exclude_list=["domain"])
    # Wavelets output basses
    output_basis_params = {"pywt_name": pywt_name, "moments": moments, "init_dilat": init_dilat, "translat": translat,
                           "dilat": dilat, "n_dilat": n_dilat, "approx_level": approx_level,
                           "add_constant": add_constant, "locs_bounds": locs_bounds_out, "domain": domain_out}
    bases_out = configs_generation.subconfigs_combinations("wavelets", output_basis_params,
                                                           exclude_list=["domain", "locs_bounds"])
    # Generate full configs
    params = {"basis_in": bases_in, "basis_out": bases_out, 'basis_rffs': bases_rffs, "regu": regu,
              "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.TripleBasisEstimator(**config) for config in configs]
    return configs, regs


def speech_fpca_3be(ker_sigma, regu, n_fpca, n_evals_fpca, domain=np.array([[0, 1]])):
    # FPCA output basis
    output_basis_params = {"n_basis": n_fpca, "input_dim": 1, "domain": domain, "n_evals": n_evals_fpca}
    output_bases = configs_generation.subconfigs_combinations("functional_pca",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    ker_sigmas = ker_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Generate full configs
    params = {"kernel": multi_ker, "basis_out": output_bases, "regu": regu, "center_output": True}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.BiBasisEstimator(**config) for config in configs]
    return configs, regs


def speech_fourier_3be(ker_sigma, regu, upper_freqs, domain=np.array([[0, 1]])):
    # Fourier output basis
    output_basis_params = {"lower_freq": 0, "upper_freq": upper_freqs, "domain": domain}
    output_bases = configs_generation.subconfigs_combinations("fourier",
                                                              output_basis_params,
                                                              exclude_list=["domain"])
    # Sum of Gaussian kernels
    ker_sigmas = ker_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    # Generate full configs
    params = {"kernel": multi_ker, "basis_out": output_bases, "regu": regu, "center_output": True}
    configs = configs_generation.configs_combinations(params)
    # Create list of regressors from that config
    regs = [triple_basis.BiBasisEstimator(**config) for config in configs]
    return configs, regs


# ############################### FKRR #################################################################################

def kernels_generator_fkrr_dti(kin_sigma, kout_sigma):
    if isinstance(kin_sigma, Iterable):
        kernels_in = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in kin_sigma]
    else:
        kernels_in = kernels.GaussianScalarKernel(kin_sigma, normalize=False)
    if isinstance(kout_sigma, Iterable):
        kernels_out = [kernels.LaplaceScalarKernel(sig, normalize=False) for sig in kout_sigma]
    else:
        kernels_out = kernels.LaplaceScalarKernel(kout_sigma, normalize=False)
    return kernels_in, kernels_out


def toy_spline_fkrr(kin_sigma, kout_sigma, regu, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_fkrr_dti(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "approx_locs": approx_locs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctional(**config) for config in configs]
    return configs, regs


def dti_fkrr(kin_sigma, kout_sigma, regu, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_fkrr_dti(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "approx_locs": approx_locs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctional(**config) for config in configs]
    return configs, regs


def kernels_generator_fkrr_speech(kin_sigma, kout_sigma):
    kin_sigmas = kin_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kin_sigmas]
    kernels_in = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    if isinstance(kout_sigma, Iterable):
        kernels_out = [kernels.LaplaceScalarKernel(sig, normalize=False) for sig in kout_sigma]
    else:
        kernels_out = kernels.LaplaceScalarKernel(kout_sigma, normalize=False)
    return kernels_in, kernels_out


def speech_fkrr(kin_sigma, kout_sigma, regu, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_fkrr_speech(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "approx_locs": approx_locs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctional(**config) for config in configs]
    return configs, regs


def speech_fkrr_eig(kin_sigma, kout_sigma, regu, neig_in, neig_out, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_fkrr_speech(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "neig_out": neig_out, "neig_in": neig_in, "approx_locs": approx_locs,
              "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctionalEigsolve(**config) for config in configs]
    return configs, regs


def speech_fkrr_eigapprox(kin_sigma, kout_sigma, regu, neig, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_fkrr_speech(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "neig": neig, "approx_locs": approx_locs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctionalEigapprox(**config) for config in configs]
    return configs, regs


def kernels_generator_gauss_fkrr_speech(kin_sigma, kout_sigma):
    kin_sigmas = kin_sigma * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kin_sigmas]
    kernels_in = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    if isinstance(kout_sigma, Iterable):
        kernels_out = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in kout_sigma]
    else:
        kernels_out = kernels.GaussianScalarKernel(kout_sigma, normalize=False)
    return kernels_in, kernels_out


def speech_fkrr_gauss(kin_sigma, kout_sigma, regu, approx_locs, center_output):
    kernels_in, kernels_out = kernels_generator_gauss_fkrr_speech(kin_sigma, kout_sigma)
    params = {"regu": regu, "kernel_in": kernels_in, "kernel_out": kernels_out,
              "approx_locs": approx_locs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params, exclude_list=["approx_locs"])
    regs = [ovkernel_ridge.SeparableOVKRidgeFunctional(**config) for config in configs]
    return configs, regs



# ############################### KE ###################################################################################

def dti_ke(kx_sigma):
    kxs = [kernels.GaussianScalarKernel(sig, normalize=False) for sig in kx_sigma]
    params = {"kernel": kxs}
    configs = configs_generation.configs_combinations(params)
    regs = [kernel_estimator.KernelEstimatorStructIn(**config) for config in configs]
    return configs, regs


def kernel_generator_ke_speech(kx_sigma):
    if isinstance(kx_sigma, Iterable):
        multi_sigs = [sig * np.ones(13) for sig in kx_sigma]
        bases_kers = [[kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in multi_sig]
                      for multi_sig in multi_sigs]
        kxs = [kernels.SumOfScalarKernel(base_ker, normalize=False) for base_ker in bases_kers]
    else:
        multi_sig = kx_sigma * np.ones(13)
        base_ker = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in multi_sig]
        kxs = kernels.SumOfScalarKernel(base_ker, normalize=False)
    return kxs


def speech_ke(kx_sigma, center_output):
    kxs = kernel_generator_ke_speech(kx_sigma)
    params = {"kernel": kxs, "center_output": center_output}
    configs = configs_generation.configs_combinations(params)
    regs = [kernel_estimator.KernelEstimatorStructIn(**config) for config in configs]
    return configs, regs
