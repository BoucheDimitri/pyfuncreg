import numpy as np
import os
import pickle
import sys
import pathlib

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent.parent.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from data import loading
from expes import generate_expes
from data import processing
from model_eval import parallel_tuning

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_3be_multi"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_3be"

N_FOLDS = 5
INPUT_INDEXING = "list"
OUTPUT_INDEXING = "discrete_general"

# Exec config
N_PROCS = 7
MIN_PROCS = None
# N_PROCS = None
# MIN_PROCS = 32

# ############################### Regressor config #####################################################################

# Output domain
DOMAIN = np.array([[0, 1]])
# Regularization parameters grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA = 40
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300

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

    # ############################# Load the data ######################################################################
    X, Y = loading.load_raw_speech_dataset(path + "/data/dataspeech/raw/")

    for i in range(N_AVERAGING):
        Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
            X, Y, shuffle_seed=seeds_data[i], n_train=300, normalize_domain=True, normalize_values=True)

        # try:
        #     key = sys.argv[1]
        # except IndexError:
        #     raise IndexError(
        #         'You need to define a vocal tract subproblem '
        #         'in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}')
        key = "LA"
        Ytrain_ext, Ytrain, Ytest_ext, Ytest \
            = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

        # Generate configs and corresponding regressors
        configs, regs = generate_expes.speech_fpca_3be(KER_SIGMA, REGU_GRID, N_FPCA, NEVALS_FPCA, DOMAIN)

        # Cross validation of the regressors
        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain_ext, Xtest, Ytest, Xpred_train=None, Ypred_train=Ytrain,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS, min_nprocs=MIN_PROCS)

        best_configs.append(best_config)
        best_results.append(best_results)
        scores_test.append(score_test)

        with open(rec_path + "/" + str(i) + "_" + key + ".pkl", "wb") as out:
            pickle.dump((best_configs, best_results, scores_test), out, pickle.HIGHEST_PROTOCOL)
