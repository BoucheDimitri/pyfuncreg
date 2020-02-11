from collections.abc import Iterable
import itertools
import numpy as np

from functional_regressors import kernels
from functional_regressors import kernel_additive
from functional_regressors import kernel_estimator
from functional_regressors import ovkernel_ridge
from functional_regressors import kernel_projection_learning as kproj_learning
from functional_regressors import triple_basis
from functional_data import basis


def expe_generator(expe_parametrizer):
    expe_queue = []
    multi_params = []
    uni_params = []
    multi_vals = []
    for key in expe_parametrizer.keys():
        if isinstance(expe_parametrizer[key], Iterable) and not isinstance(expe_parametrizer[key], str):
            multi_params.append(key)
            multi_vals.append(expe_parametrizer[key])
        else:
            uni_params.append(key)
    for params_set in itertools.product(*multi_vals):
        new_dict = {key: expe_parametrizer[key] for key in uni_params}
        count = 0
        for param in params_set:
            new_dict[multi_params[count]] = param
            count += 1
        expe_queue.append(new_dict)
    return expe_queue


# ############################# Kernel additive model (KAM) ############################################################
def create_kam_dti(expe_dict, n_evals_in, n_evals_out, domain_in, domain_out, n_evals_fpca):
    kerlocs_in = kernels.GaussianScalarKernel(stdev=expe_dict["kx"], normalize=False)
    kerlocs_out = kernels.GaussianScalarKernel(stdev=expe_dict["ky"], normalize=False)
    kerevals = kernels.GaussianScalarKernel(stdev=expe_dict["keval"], normalize=False)
    reg = kernel_additive.KernelAdditiveModel(expe_dict["regu"], kerlocs_in, kerlocs_out,
                                              kerevals, n_evals_in, n_evals_out, expe_dict["nfpca"],
                                              domain_in, domain_out, n_evals_fpca)
    return reg


# ############################# Kernel estimator (KE) ##################################################################
def create_ke_dti(expe_dict):
    gauss_ker = kernels.GaussianScalarKernel(stdev=np.sqrt(expe_dict["window"]), normalize=False)
    reg = kernel_estimator.KernelEstimatorFunc(gauss_ker)
    return reg


def create_ke_speech(expe_dict):
    ker_sigmas = expe_dict["ker_sigma"] * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    reg = kernel_estimator.KernelEstimatorStructIn(multi_ker, center_output=expe_dict["center_output"])
    return reg


# ############################# Functional kernel ridge regression (FKR) ###############################################
def create_fkr_dti(expe_dict, approx_locs):
    gauss_ker = kernels.GaussianScalarKernel(expe_dict["ker_sigma"], normalize=False)
    lap_ker = kernels.LaplaceScalarKernel(expe_dict["ky"], normalize=False)
    reg = ovkernel_ridge.SeparableOVKRidgeFunctional(expe_dict["regu"], gauss_ker, lap_ker,
                                                     approx_locs, expe_dict["center_outputs"])
    return reg


def create_fkr_speech(expe_dict, approx_locs):
    ker_sigmas = expe_dict["ker_sigma"] * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    lap_ker = kernels.LaplaceScalarKernel(expe_dict["ky"], normalize=False)
    reg = ovkernel_ridge.SeparableOVKRidgeFunctional(expe_dict["regu"], multi_ker, lap_ker,
                                                     approx_locs, expe_dict["center_outputs"])
    return reg


# ############################# Kernel-based projection learning (KPL) #################################################
def create_kpl_dti(expe_dict, domain_out, domain_out_pad, pad_width):
    func_dict = basis.MultiscaleCompactlySupported(domain_out, domain_out_pad,
                                                   pywt_name=expe_dict["pywt_name"],
                                                   moments=expe_dict["moments"],
                                                   init_dilat=expe_dict["init_dilat"],
                                                   dilat=expe_dict["dilat"],
                                                   n_dilat=expe_dict["n_dilat"],
                                                   translat=expe_dict["translat"],
                                                   approx_level=6,
                                                   add_constant=True)
    if expe_dict["penalize_freqs"] is not None:
        if expe_dict["penalize_freqs"] == "Linear":
            freqs_penalization = kproj_learning.wavelet_freqs_penalization(func_dict, mode=expe_dict["penalize_freqs"])
        else:
            freqs_penalization = kproj_learning.wavelet_freqs_penalization(func_dict, mode=None,
                                                                           decrease_base=expe_dict["penalize_freqs"])
        B = np.diag(freqs_penalization)
    else:
        B = np.eye(func_dict.n_basis)
    gauss_ker = kernels.GaussianScalarKernel(expe_dict["ker_sigma"], normalize=False)
    non_padded_index = (pad_width[1][0], 55 + pad_width[1][0])
    reg = kproj_learning.KPLExact(gauss_ker, B, func_dict, expe_dict["regu"],
                                  non_padded_index=non_padded_index, center_output=expe_dict["center_outputs"])
    # bfgs = first_order.ScipySolver(maxit=config.MAXIT, tol=config.TOL, method="L-BFGS-B")
    # non_padded_index = (config.PAD_WIDTH[1][0], 55 + config.PAD_WIDTH[1][0])
    # reg = dictout_ovkreg.DictOutOVKRidge(gauss_ker, B, func_dict, expe_dict["regu"], bfgs,
    #                                      non_padded_index=non_padded_index,
    #                                      center_outputs=expe_dict["center_outputs"])
    return reg


def create_kpl_speech(expe_dict, nevals_fpca):
    ker_sigmas = expe_dict["ker_sigma"] * np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    reg = kproj_learning.KPLExactFPCA(multi_ker, expe_dict["regu"],
                                      expe_dict["n_fpca"], nevals_fpca=nevals_fpca,
                                      penalize_eigvals=expe_dict["penalize_eigvals"],
                                      penalize_pow=expe_dict["penalize_pow"],
                                      center_output=expe_dict["center_output"])
    return reg


# ############################# Triple basis estimator (3BE) ###########################################################
def create_3be_dti(expe_dict, domain_in, domain_out, nrffs, rffs_seed, pad_width):
    basis_in = basis.FourierBasis((0, expe_dict["max_freq_in"]), domain_in)
    basis_out = basis.FourierBasis((0, expe_dict["max_freq_out"]), domain_out)
    rffs = basis.RandomFourierFeatures(nrffs, domain=domain_in,
                                       bandwidth=expe_dict["ker_sigma"],
                                       input_dim=basis_in.n_basis, seed=rffs_seed)
    non_padded_index = (pad_width[1][0], 55 + pad_width[1][0])
    reg = triple_basis.TripleBasisEstimator(basis_in,rffs, basis_out, expe_dict["regu"],
                                            non_padded_index=non_padded_index,
                                            center_output=expe_dict["center_outputs"])
    return reg


def create_3be_speech(expe_dict, nevals_fpca):
    ker_sigmas = expe_dict["ker_sigma"] * np.ones(13)
    kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(kers, normalize=False)
    reg = triple_basis.BiBasisEstimatorFpca(multi_ker, expe_dict["regu"], expe_dict["nfpca"],
                                            center_output=expe_dict["center_output"], nevals_fpca=nevals_fpca)
    return reg
