import os
import numpy as np
import importlib

from data import loading
from data import processing
from functional_regressors import kernel_projection_learning as kproj
from functional_regressors import kernels
from functional_regressors import regularization
from functional_data import basis
from misc import model_eval
from functional_data import fpca

importlib.reload(kproj)
importlib.reload(fpca)
importlib.reload(loading)

shuffle_seed = 784
path = os.getcwd()
n_train = 70
signal_ext = ("symmetric", (1, 1))
center_output = True
domain_out = np.array([[0, 1]])
locs_bounds = np.array([[0 - signal_ext[1][0], 1 + signal_ext[1][1]]])
decrease_base = 1.2

Xtrain, Ytrain, Xtest, Ytest = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=shuffle_seed, n_train=n_train)

gauss_ker = kernels.GaussianScalarKernel(0.9, normalize=False)
output_basis_config = output_basis_params = {"pywt_name": "db", "moments": 2,
                                             "init_dilat": 1, "translat": 1,
                                             "approx_level": 6, "add_constant": True,
                                             "locs_bounds": locs_bounds, "domain": domain_out}
output_basis = ("wavelets", output_basis_config)
output_matrix = regularization.WaveletsPow(decrease_base)

regu = 1e-3

test_kpl = kproj.SeperableKPL(gauss_ker, output_matrix, output_basis, regu, center_output, signal_ext)

Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

test_kpl.fit(Xtrain, Ytrain)



preds = test_kpl.predict_evaluate_diff_locs(Xtest, Ytest[0])
score_test = model_eval.mean_squared_error(preds, Ytest[1])

test_fpca1 = fpca.FunctionalPCA(domain_out, 30)
spaces, outputs = test_fpca1.fit(Ytrain[0], Ytrain[1])

test_fpca = basis.FPCABasis(**output_basis_config)
test_fpca.fit(Ytrain[0], Ytrain[1])
test_comp_mat = test_fpca.compute_matrix(Ytrain[0][0])