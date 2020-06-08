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
from model_eval import perf_timing

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_3be_timer"
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
SIGNAL_EXT_INPUT = ("symmetric", (1, 1))
SIGNAL_EXT_OUTPUT = ("symmetric", (1, 1))
DOMAIN = np.array([[0, 1]])
LOCS_BOUNDS_IN = np.array([[0 - SIGNAL_EXT_INPUT[1][0], 1 + SIGNAL_EXT_INPUT[1][1]]])
LOCS_BOUNDS_OUT = np.array([[0 - SIGNAL_EXT_OUTPUT[1][0], 1 + SIGNAL_EXT_OUTPUT[1][1]]])

# MOMENTS = [2, 3]
# PYWT_NAME = ["db"]
# N_DILATS = [4, 5]
# MOMENTS_IN = [2, 3]
# PYWT_NAME_IN = ["db"]
# N_DILATS_IN = [4, 5]
# BASIS_DICT_IN = {"pywt_name_in": PYWT_NAME_IN, "moments_in": MOMENTS_IN, "n_dilat_in": N_DILATS_IN,
#                  "add_constant_in": True, "domain_in": DOMAIN, "locs_bounds_in": LOCS_BOUNDS_IN}
#
# MOMENTS_OUT = [2, 3]
# PYWT_NAME_OUT = ["db"]
# N_DILATS_OUT = [4, 5]
# BASIS_DICT_OUT = {"pywt_name_out": PYWT_NAME_OUT, "moments_out": MOMENTS_OUT, "n_dilat_out": N_DILATS_OUT,
#                   "add_constant_out": True, "domain_out": DOMAIN, "locs_bounds_out": LOCS_BOUNDS_OUT}

MOMENTS_IN = [2, 3]
PYWT_NAME_IN = ["db"]
N_DILATS_IN = [4]
BASIS_DICT_IN = {"pywt_name_in": PYWT_NAME_IN, "moments_in": MOMENTS_IN, "n_dilat_in": N_DILATS_IN,
                 "add_constant_in": True, "domain_in": DOMAIN, "locs_bounds_in": LOCS_BOUNDS_IN}

MOMENTS_OUT = [2, 3]
PYWT_NAME_OUT = ["db"]
N_DILATS_OUT = [4]
BASIS_DICT_OUT = {"pywt_name_out": PYWT_NAME_OUT, "moments_out": MOMENTS_OUT, "n_dilat_out": N_DILATS_OUT,
                  "add_constant_out": True, "domain_out": DOMAIN, "locs_bounds_out": LOCS_BOUNDS_OUT}

# Number of random fourier features
N_RFFS = 145
# Seed for the random fourier features
RFFS_SEED = 567
# Regularization grid
# REGU_GRID = list(np.geomspace(1e-8, 1, 100))
REGU_GRID = list(np.geomspace(1e-8, 1, 25))
# REGU_GRID = [1e-1, 1]
# Standard deviation grid for input kernel
KER_SIGMA = [1, 2.5, 5, 10]
# KER_SIGMA = [1, 10, 20]
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

    results = list()

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

        configs, regs = generate_expes.dti_3be_wavs(KER_SIGMA, REGU_GRID, CENTER_OUTPUT,
                                                    N_RFFS, RFFS_SEED, **BASIS_DICT_IN, **BASIS_DICT_OUT)

        result = perf_timing.parallel_perf_counter(regs, Xtrain_extended, Ytrain_extended,
                                                   Xtest, Ytest, n_procs=N_PROCS,
                                                   min_nprocs=MIN_PROCS)
        results.append(result)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
        print(i)
    print(results)