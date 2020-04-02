import numpy as np
import os
import sys
import pathlib
import pickle

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from data import loading, processing
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd1
from model_eval import parallel_tuning

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_3be_multi"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_3be"

INPUT_INDEXING = "discrete_general"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5

# Exec config
# N_PROCS = 7
# MIN_PROCS = None
N_PROCS = None
MIN_PROCS = 32
# MIN_PROCS = None

# ############################### Regressor config #####################################################################
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
SIGNAL_EXT_INPUT = ("symmetric", (0, 0))
SIGNAL_EXT_OUTPUT = ("symmetric", (0, 0))
# Output domain
DOMAIN = np.array([[0, 1]])
# Number of random fourier features
N_RFFS = 300
# Seed for the random fourier features
RFFS_SEED = 567
# Regularization grid
REGU_GRID = list(np.geomspace(1e-8, 1, 100))
# REGU_GRID = [1e-1, 1]
# Standard deviation grid for input kernel
KER_SIGMA = [15, 20, 25, 30, 40]
# KER_SIGMA = 20
# Maximum frequency to include for input and output
FREQS_IN_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
FREQS_OUT_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
# FREQS_IN_GRID = [20, 25]
# FREQS_OUT_GRID = [5, 10]
CENTER_OUTPUT = True

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)

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

    scores_test, best_results, best_configs = list(), list(), list()

    for i in range(N_AVERAGING):
        # ############################# Load the data ##################################################################
        cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=seeds_data[i])
        Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)
        # Extend data
        Xtrain_extended = disc_fd1.extend_signal_samelocs(
            Xtrain[0][0], Xtrain[1], mode=SIGNAL_EXT_INPUT[0], repeats=SIGNAL_EXT_INPUT[1])
        Ytrain_extended = disc_fd1.extend_signal_samelocs(
            Ytrain[0][0], Ytrain[1], mode=SIGNAL_EXT_OUTPUT[0], repeats=SIGNAL_EXT_OUTPUT[1])
        # Convert testing output data to discrete general form
        Ytest = disc_fd1.to_discrete_general(*Ytest)

        configs, regs = generate_expes.dti_3be_fourier(KER_SIGMA, REGU_GRID, CENTER_OUTPUT, FREQS_IN_GRID,
                                                       FREQS_OUT_GRID, N_RFFS, RFFS_SEED, DOMAIN, DOMAIN)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain_extended, Ytrain_extended, Xtest, Ytest, Xpred_train=Xtrain, Ypred_train=Ytrain,
            configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS, min_nprocs=MIN_PROCS)
        best_configs.append(best_config)
        best_results.append(best_results)
        scores_test.append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((best_configs, best_results, scores_test), out, pickle.HIGHEST_PROTOCOL)
