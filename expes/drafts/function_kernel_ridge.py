from data import processing
from data import loading
from data import degradation
import os
from functional_regressors import kernels
from functional_regressors import ovkernel_ridge
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import time

X, Y = loading.load_raw_speech_dataset(os.getcwd() + "/data/dataspeech/raw/")
key = "VEL"

Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
    X, Y, shuffle_seed=543, n_train=300, normalize_domain=True, normalize_values=True)
Ytrain_ext, Ytrain, Ytest_ext, Ytest = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

kerout = kernels.LaplaceScalarKernel(0.1, False)
kin_sigmas = np.ones(13)
gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in kin_sigmas]
kerin = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
approx_locs = Ytrain_ext[0][0]
neig_out = 20
neig_in = 50
regu = 1e-8

reg = ovkernel_ridge.SeparableOVKRidgeFunctionalEigsolve(regu, kerin, kerout, neig_in, neig_out,
                                                         approx_locs, center_output=True)
#
# u, v = np.linalg.eigh(reg.Kout)
#
# Kin = kerin(Xtrain, Xtrain)
# uin, vin = np.linalg.eigh(Kin)

cpu_time = reg.fit(Xtrain, Ytrain_ext, return_cputime=True)