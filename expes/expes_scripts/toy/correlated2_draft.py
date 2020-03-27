import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
import pathlib
import importlib
import time

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent.parent.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from data import toy_data_spline
from data import degradation
from expes import generate_expes
from model_eval import parallel_tuning
from functional_regressors import kernel_projection_learning as kpl

importlib.reload(kpl)

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "toy_correlated2"
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
# N_PROCS = None
# MIN_PROCS = 32
N_PROCS = 7
MIN_PROCS = None

# ############################### Regressor config #####################################################################
# REGU = np.geomspace(1e-8, 1, 300)
REGU = np.geomspace(1e-8, 1, 100)
# N_SAMPLES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500]
N_SAMPLES = [200]
# TASKS_CORREL = [0.1, 0.2]
# TASKS_CORREL = np.arange(0.1, 1, 0.05)
# TASKS_CORREL = toy_data_spline.estimate_correlation()
TASKS_CORREL = 0.7
# TASKS_CORREL = 0L
KER_SIGMA = 20

NOISE_INPUT = 0
NOISE_OUTPUT = 0
# NOISE_INPUT = 0.1
# NOISE_OUTPUT = 0.04
SEED_INPUT = 768
SEED_OUTPUT = 456

# ############################# Load the data ######################################################################
configs_corr, regs_corr = generate_expes.toy_spline_kpl_corr2(KER_SIGMA, REGU, TASKS_CORREL)
configs, regs = generate_expes.toy_spline_kpl(KER_SIGMA, REGU)

scores_test_corr = []
scores_test = []

start = time.perf_counter()
for n_samples in N_SAMPLES:
    Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data_correlated2(n_samples)
    Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, SEED_INPUT)
    Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, SEED_OUTPUT)
    best_config_corr, best_result_corr, score_test_corr = parallel_tuning.parallel_tuning(
        regs_corr, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs_corr, n_folds=N_FOLDS, n_procs=N_PROCS,
        min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
    best_config, best_result, score_test = parallel_tuning.parallel_tuning(
        regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
        min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
    scores_test_corr.append(score_test_corr)
    scores_test.append(score_test)
end = time.perf_counter()
print(end - start)


# regtest = regs[10]
# regtest.fit(Xtrain, Ytrain)




best_reg_corr = kpl.SeperableKPL(**best_config_corr)
best_reg_corr.fit(Xtrain, Ytrain)
preds_corr = best_reg_corr.predict_evaluate_diff_locs(Xtrain, Ytrain[0])

best_config_normal = best_config_corr.copy()
best_config_normal["regu"] = best_config["regu"]
best_config_normal["B"] = np.eye(best_config_normal["basis_out"].n_basis)
best_reg = kpl.SeperableKPL(**best_config_normal)
best_reg.fit(Xtrain, Ytrain)
preds = best_reg.predict_evaluate_diff_locs(Xtrain, Ytrain[0])


fig, axes = plt.subplots(nrows=4, ncols=4)
for i in range(4):
    for j in range(4):
        # axes[i, j].plot(Ytrain[0][i + 4 * j], preds_corr[i + 4 * j], label="predicted")
        axes[i, j].plot(Ytrain[0][i + 4 * j], preds[i + 4 * j], label="predicted")
        axes[i, j].plot(Ytrain[0][i + 4 * j], Ytrain[1][i + 4 * j], label="true")


preds_corr_test = best_reg_corr.predict_evaluate_diff_locs(Xtest, Ytest[0])
preds_test = best_reg.predict_evaluate_diff_locs(Xtest, Ytest[0])

fig, axes = plt.subplots(nrows=4, ncols=4)
for i in range(4):
    for j in range(4):
        # axes[i, j].plot(Ytest[0][i + 4 * j], preds_corr_test[i + 4 * j], label="predicted_corr")
        axes[i, j].plot(Ytest[0][i + 4 * j], preds_test[i + 4 * j], label="predicted")
        axes[i, j].plot(Ytest[0][i + 4 * j], Ytest[1][i + 4 * j], label="true")



import time

start = time.perf_counter()
best_reg_corr.fit(Xtrain, Ytrain)
end = time.perf_counter()
print(end - start)

start = time.perf_counter()
best_reg_corr_svd.fit(Xtrain, Ytrain)
end = time.perf_counter()
print(end - start)


