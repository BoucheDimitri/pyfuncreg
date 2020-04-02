import numpy as np
import os
import pickle
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from data import toy_data_spline
from data import degradation
from expes import generate_expes
from model_eval import parallel_tuning

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "toy_correlated_multi"
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 2
N_PROCS = None
MIN_PROCS = 32
# N_PROCS = 7
# MIN_PROCS = None

# ############################### Regressor config #####################################################################
REGU = np.geomspace(1e-11, 1e2, 500)
# REGU = np.geomspace(1e-8, 1, 100)
N_SAMPLES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]
# N_SAMPLES = [5, 10, 15, 20, 25, 50, 100]
# N_SAMPLES = [10, 20, 30, 40, 50, 60, 70, 80, 100]
# TASKS_CORREL = [0.1, 0.2]
TASKS_CORREL = np.arange(0, 1, 0.05)
# TASKS_CORREL = toy_data_spline.estimate_correlation()
# TASKS_CORREL = 0L
KER_SIGMA = 10

# Noise parameters
NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784
SEED_INPUT = 768
SEED_OUTPUT = 456

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_INPUT)
seeds_noise_in = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_OUTPUT)
seeds_noise_out = np.random.randint(100, 2000, N_AVERAGING)

if __name__ == '__main__':

    # ############################# Create folder for recording ########################################################
    try:
        os.mkdir(path + "/outputs")
    except FileExistsError:
        pass
    try:
        os.mkdir(path + "/outputs/" + OUTPUT_FOLDER)
    except FileExistsError:
        pass
    rec_path = path + "/outputs/" + OUTPUT_FOLDER

    # ############################# Load the data ######################################################################
    configs_corr, regs_corr = generate_expes.toy_spline_kpl_corr2(KER_SIGMA, REGU, TASKS_CORREL)
    configs, regs = generate_expes.toy_spline_kpl2(KER_SIGMA, REGU)

    scores_dicts = []
    scores_dicts_corr = []

    for i in range(N_AVERAGING):
        scores_dicts.append({})
        scores_dicts_corr.append({})
        for n_samples in N_SAMPLES:
            scores_dicts[i][n_samples] = []
            scores_dicts_corr[i][n_samples] = []
            Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data_correlated2(n_samples, seed=seeds_data[i])
            Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
            Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, seeds_noise_out[i])
            best_config_corr, best_result_corr, score_test_corr = parallel_tuning.parallel_tuning(
                regs_corr, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs_corr, n_folds=N_FOLDS, n_procs=N_PROCS,
                min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
            best_config, best_result, score_test = parallel_tuning.parallel_tuning(
                regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
                min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
            scores_dicts[i][n_samples].append(score_test)
            scores_dicts_corr[i][n_samples].append(score_test_corr)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((scores_dicts, scores_dicts_corr), out, pickle.HIGHEST_PROTOCOL)

    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((scores_dicts, scores_dicts_corr), out, pickle.HIGHEST_PROTOCOL)



