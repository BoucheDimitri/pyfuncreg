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
from expes import generate_expes
from model_eval import metrics, perf_timing
from data import loading, processing
from functional_data import discrete_functional_data as disc_fd

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_fkrr_timer"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_fkrr"

# Exec config
# N_PROCS = 7
# MIN_PROCS = None
N_PROCS = None
MIN_PROCS = 32
# MIN_PROCS = None

N_FOLDS = 5
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"

# ############################### Regressor config #####################################################################
# Domain
DOMAIN_OUT = np.array([[0, 1]])
# Regularization parameter grid
REGU = np.geomspace(1e-8, 1, 25)
# REGU_GRID = 0.00271858824273294
# REGU = [1e-4, 1e-3]
# Output kernel bandwidth grid
# KY_GRID = [0.01, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
# KOUT_SIGMA = [0.01, 0.1]
KOUT_SIGMA = [0.01, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0]
# Input kernel standard deviation
KIN_SIGMA = 0.9
# Location used for discrete approximation
APPROX_LOCS = np.linspace(0, 1, 100)
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
        # Convert testing output data to discrete general form
        Ytest = disc_fd.to_discrete_general(*Ytest)
        # Put input data in array form
        Xtrain = np.array(Xtrain[1]).squeeze()
        Xtest = np.array(Xtest[1]).squeeze()
        configs, regs = generate_expes.dti_fkrr(KIN_SIGMA, KOUT_SIGMA, REGU, APPROX_LOCS, CENTER_OUTPUT)

        result = perf_timing.parallel_perf_counter(regs, Xtrain, Ytrain, Xtest, Ytest, n_procs=N_PROCS,
                                                   min_nprocs=MIN_PROCS)
        results.append(result)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
        print(i)
    print(results)