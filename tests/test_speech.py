import os
import numpy as np
import importlib
import sys
import pathlib
import matplotlib.pyplot as plt
from time import perf_counter

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

from data import loading
from functional_regressors import kernels
from functional_regressors import kernel_projection_learning as kproj
from functional_regressors import regularization
from functional_data import fpca
from functional_data import smoothing
from misc import model_eval

importlib.reload(fpca)

domain_out = np.array([[0, 1]])


Xtrain, Ytrain_full, Xtest, Ytest_full = loading.load_processed_speech_dataset(path + "/data/dataspeech/processed/")

key = "LA"

Ytrain = Ytrain_full[key]
Ytest = Ytest_full[key]

# Ytrain = ([np.expand_dims(Ytrain[0][i], axis=1) for i in range(len(Ytrain[0]))], Ytrain[1])

#
# test_fpca1 = fpca.FunctionalPCA(domain_out, 500)
# test_fpca1.fit(Ytrain[0], Ytrain[1])
# test_pred = test_fpca1.predict(Ytrain[0][0])
#
# plt.plot(test_pred[0])
#
# smoother = smoothing.LinearInterpSmoother()
# smoother.fit(Ytrain[0], Ytrain[1])
# Yfuncs = smoother.get_functions()
#
# test_fpca2 = fpca.FunctionalPCA(domain_out, 30)
# test_fpca2.fit(Yfuncs)


ker_sigmas = 1 * np.ones(13)
gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)

output_matrix = regularization.Eye()
output_basis_config = {"n_basis": 40, "input_dim": 1, "domain": domain_out, "n_evals": 300}
output_basis = ('functional_pca', output_basis_config)

regu = 1e-10

test_kproj = kproj.SeperableKPL(multi_ker, output_matrix, output_basis, regu)

# start = perf_counter()
test_kproj.fit(Xtrain, Ytrain)
# end = perf_counter()
# print(end - start)

preds = test_kproj.predict_evaluate_diff_locs(Xtest, Ytest[0])
score_test = model_eval.mean_squared_error(preds, Ytest[1])
