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

shuffle_seed = 784
path = os.getcwd()
n_train = 70
pad_width = ((0, 0), (0, 0))
domain_out = np.array([[0, 1]])

cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=shuffle_seed)
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca.copy(), rcst.copy(),
                                                              n_train=n_train, normalize01=True,
                                                              pad_width_output=pad_width)

gauss_ker = kernels.GaussianScalarKernel(0.9, normalize=False)
output_basis_config = {"domain": domain_out, "input_dim": 1, "n_basis": 10, "n_evals": 40}
output_basis = ("functional_pca", output_basis_config)
output_matrix = regularization.Eye()

regu = 1e-3

test_kpl = kproj.SeperableKPL(gauss_ker, output_matrix, output_basis, regu)

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