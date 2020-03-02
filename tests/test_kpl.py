import os
import numpy as np
import importlib

from data import loading
from data import processing
from functional_data import discrete_functional_data as disc_fd
from functional_regressors import kernel_projection_learning as kproj
from functional_regressors import kernels
from functional_regressors import regularization
from functional_data import basis
from misc import model_eval
from functional_data import fpca
from functional_regressors.DEPRECATED import kernel_projection_learning as kproj_dp

importlib.reload(kproj)
importlib.reload(fpca)
importlib.reload(loading)
importlib.reload(disc_fd)
importlib.reload(processing)

shuffle_seed = 784
path = os.getcwd()
n_train = 70
signal_ext = ("symmetric", (1, 1))
center_output = True
domain_out = np.array([[0, 1]])
locs_bounds = np.array([[0 - signal_ext[1][0], 1 + signal_ext[1][1]]])
decrease_base = 1.2

cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=shuffle_seed)
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)

gauss_ker = kernels.GaussianScalarKernel(0.9, normalize=False)
output_basis_config = output_basis_params = {"pywt_name": "db", "moments": 2,
                                             "init_dilat": 1, "translat": 1, "dilat": 2,
                                             "approx_level": 6, "add_constant": True,
                                             "locs_bounds": locs_bounds, "domain": domain_out}
output_basis = ("wavelets", output_basis_config)
output_matrix = regularization.WaveletsPow(decrease_base)

regu = 1e-3


# # TEST DEPREC ########################################################################################################
Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

Ytrain_wrapped = disc_fd.wrap_functional_data(Ytrain, key='discrete_samelocs_regular_1d')
Ytrain_wrapped = Ytrain_wrapped.extended_version(repeats=signal_ext[1])

mean_func = Ytrain_wrapped.mean_func()
test_mean = mean_func(Ytrain[0])


Ytrain = Ytrain_wrapped.discrete_general()

Ytest_wrapped = disc_fd.wrap_functional_data(Ytest, key='discrete_samelocs_regular_1d')
Ytest = Ytest_wrapped.discrete_general()

test_kpl = kproj_dp.SeperableKPL(kernel_scalar=gauss_ker, B=output_matrix, output_basis=output_basis, regu=regu,
                                 center_output=False)

test_kpl.fit(Xtrain, Ytrain)

preds = test_kpl.predict_evaluate_diff_locs(Xtest, Ytest[0])
score_test = model_eval.mean_squared_error(preds, Ytest[1])


# #########" TEST NEW
test_kpl = kproj.SeperableKPL(kernel_scalar=gauss_ker, B=output_matrix, output_basis=output_basis, regu=regu,
                              center_output=center_output, signal_ext=signal_ext)

Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

# Ywrapped = disc_fd.wrap_functional_data(Ytrain, key='discrete_samelocs_regular_1d')
# # Memorize mean function before signal extension
# if center_output:
#     Ymean_func = Ywrapped.mean_func()
# # Extends the signal if relevant
# Ywrapped = Ywrapped.extended_version(signal_ext[0], signal_ext[1])
# # Center with extended signal if relevant
# if center_output:
#     Ycentered = Ywrapped.centered_discrete_general()
# else:
#     Ycentered = Ywrapped.discrete_general()

# Ycentered = test_kpl.fit(Xtrain, Ytrain)

test_kpl.fit(Xtrain, Ytrain)

Ytest_wrapped = disc_fd.wrap_functional_data(Ytest, key='discrete_samelocs_regular_1d')
Ytest = Ytest_wrapped.discrete_general()

preds = test_kpl.predict_evaluate_diff_locs(Xtest, Ytest[0])
score_test = model_eval.mean_squared_error(preds, Ytest[1])

test_fpca1 = fpca.FunctionalPCA(domain_out, 30)
spaces, outputs = test_fpca1.fit(Ytrain[0], Ytrain[1])

test_fpca = basis.FPCABasis(**output_basis_config)
test_fpca.fit(Ytrain[0], Ytrain[1])
test_comp_mat = test_fpca.compute_matrix(Ytrain[0][0])