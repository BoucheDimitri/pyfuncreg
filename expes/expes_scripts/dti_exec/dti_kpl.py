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
from model_eval import parallel_tuning
from data import loading
from data import processing
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kpl_multi_modif"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER

# Number of processors
# N_PROCS = 7
N_PROCS = None
MIN_PROCS = 32
# MIN_PROCS = None


N_TRAIN = 70
N_FOLDS = 5
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"

# ############################### Regressor config #####################################################################
# Signal extension method
SIGNAL_EXT = ("symmetric", (1, 1))
# SIGNAL_EXT = ("symmetric", (2, 2))
# SIGNAL_EXT = ("symmetric", (0, 0))
CENTER_OUTPUT = True
DOMAIN_OUT = np.array([[0, 1]])
LOCS_BOUNDS = np.array([[0 - SIGNAL_EXT[1][0], 1 + SIGNAL_EXT[1][1]]])
DECREASE_BASE = np.arange(1, 2, 0.05)
# DECREASE_BASE = np.arange(1, 1.6, 0.05)
MOMENTS = [2, 3]
PYWT_NAME = ["db"]
N_DILATS = [4, 5]
# MOMENTS = [2]
# PYWT_NAME = ["db"]
# N_DILATS = [4]
BASIS_DICT = {"pywt_name": PYWT_NAME, "moments": MOMENTS, "n_dilat": N_DILATS, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
              "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}
# Standard deviation parameter for the input kernel
KER_SIGMA = 0.9
# Regularization grid
REGUS = np.geomspace(1e-8, 1, 100)
# REGUS = [1e-3, 1e-4]


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
        Ytrain_extended = disc_fd.extend_signal_samelocs(
            Ytrain[0][0], Ytrain[1], mode=SIGNAL_EXT[0], repeats=SIGNAL_EXT[1])
        # Convert testing output data to discrete general form
        Ytest = disc_fd.to_discrete_general(*Ytest)

        # Put input data in array form
        Xtrain = np.array(Xtrain[1]).squeeze()
        Xtest = np.array(Xtest[1]).squeeze()

        # ############################# Full cross-validation experiment ###############################################
        # Generate configurations and regressors
        configs, regs = generate_expes.dti_wavs_kpl(KER_SIGMA, REGUS, center_output=CENTER_OUTPUT,
                                                    decrease_base=DECREASE_BASE, **BASIS_DICT)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain_extended, Xtest, Ytest, Xpred_train=None, Ypred_train=Ytrain,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING, min_nprocs=MIN_PROCS,
            configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        best_configs.append(best_config)
        best_results.append(best_result)
        scores_test.append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((best_configs, best_results, scores_test), out, pickle.HIGHEST_PROTOCOL)