import numpy as np
import os
import sys
import pathlib
from data import loading, processing

# Execution path
path = os.getcwd()

# Local imports
from expes import generate_expes, run_expes
from model_eval import metrics

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/raw/"

# Indexing
INPUT_INDEXING = "list"
OUTPUT_INDEXING = "discrete_general"
# Number of folds
N_FOLDS = 5

# Exec config
# N_PROCS = 7
# MIN_PROCS = None
N_PROCS = None
MIN_PROCS = 32

# ############################### KPL #####################################################################
# Output domain
DOMAIN = np.array([[0, 1]])

# Regularization parameters grid
REGU_GRID = list(np.geomspace(1e-10, 1e-3, 50))
# REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA = [30, 40]
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Decrease base
# DECREASE_BASE = np.arange(1, 1.6, 0.1)
DECREASE_BASE = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)

# Generate configurations and corresponding regressors
configs_kpl, regs_kpl = generate_expes.speech_fpca_penpow_kpl(KER_SIGMA, REGU_GRID, N_FPCA,
                                                              NEVALS_FPCA, DECREASE_BASE, DOMAIN)



# ############################### 3BE #####################################################################

# Output domain
DOMAIN = np.array([[0, 1]])
# Regularization parameters grid
REGU_GRID = list(np.geomspace(1e-10, 1e-3, 50))
# REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA = [30, 40]
# N_FPCA = [20]
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300

# Generate configurations and corresponding regressors
configs_3be, regs_3be = generate_expes.speech_fpca_3be(KER_SIGMA, REGU_GRID, N_FPCA, NEVALS_FPCA, DOMAIN)


# ############################################ FIT ####################################################################
key = "LP"

X, Y = loading.load_raw_speech_dataset(DATA_PATH)
Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
    X, Y, shuffle_seed=784, n_train=300, normalize_domain=True, normalize_values=True)
Ytrain_ext, Ytrain, Ytest_ext, Ytest \
    = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

regs_3be[0].fit(Xtrain, Ytrain_ext)
regs_kpl[0].fit(Xtrain, Ytrain_ext)

test_3be, test_kpl = regs_3be[0], regs_kpl[0]

space = np.linspace(0, 1, 100)

# Score on train set
pred_kpl, pred_3be = test_kpl.predict_evaluate_diff_locs(Xtrain, Ytrain[0]), test_3be.predict_evaluate_diff_locs(Xtrain, Ytrain[0])
score_kpl, score_3be = metrics.mse(pred_kpl, Ytrain[1]), metrics.mse(pred_3be, Ytrain[1])


fpca_kpl = test_kpl.basis_out.compute_matrix(space)
fpca_3be = test_3be.basis_out.compute_matrix(space)

for i in range(30):
    test_3be.regressors[i].reg.dual_coef_ = test_kpl.ovkridge.alpha[:, i]
    # test_3be.regressors[i].reg.dual_coef_ = np.zeros(300)

pred_kpl, pred_3be = test_kpl.predict_evaluate_diff_locs(Xtrain, Ytrain[0]), test_3be.predict_evaluate_diff_locs(Xtrain, Ytrain[0])
score_kpl, score_3be = metrics.mse(pred_kpl, Ytrain[1]), metrics.mse(pred_3be, Ytrain[1])

# Check if coefs are the same
pred_coefs_kpl, pred_coefs_3be = test_kpl.predict(Xtrain), test_3be.predict(Xtrain)

#
pred_kpl, pred_3be = test_kpl.predict_evaluate_diff_locs(Xtest, Ytest[0]), test_3be.predict_evaluate_diff_locs(Xtest, Ytest[0])
score_kpl, score_3be = metrics.mse(pred_kpl, Ytest[1]), metrics.mse(pred_3be, Ytest[1])


pred_kpl, pred_3be = test_kpl.predict_evaluate_diff_locs(Xtest, Ytest_ext[0]), test_3be.predict_evaluate_diff_locs(Xtest, Ytest_ext[0])
score_kpl, score_3be = metrics.mse(pred_kpl, Ytest_ext[1]), metrics.mse(pred_3be, Ytest_ext[1])