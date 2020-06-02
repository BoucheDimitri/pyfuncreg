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
from model_eval import perf_timing
from data import loading
from data import processing
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd1

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kam_timer"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
# Number of processors
# N_PROCS = 7
N_PROCS = None
MIN_PROCS = 32
# MIN_PROCS = None

N_TRAIN = 70
N_FOLDS = 5

# ############################### Regressor config #####################################################################
# REGU = np.geomspace(1e-8, 1, 50)
REGU = np.geomspace(1e-8, 1, 25)
N_FPCA = [20]
KIN_SIGMA = [0.1]
KOUT_SIGMA = [0.1]
KEVAL_SIGMA = [0.1]
# KIN_SIGMA = [0.01, 0.05, 0.1]
# KOUT_SIGMA = [0.01, 0.05, 0.1]
# KEVAL_SIGMA = [0.03, 0.06, 0.1]
# N_FPCA = [10, 20, 30]
# REGU= [1e-4, 1e-5]
# KIN_SIGMA = 0.01
# KOUT_SIGMA = 0.01
# KEVAL_SIGMA = 0.03
DOMAIN = np.array([[0, 1]])
N_EVALS_IN = 100
N_EVALS_OUT = 100
N_EVALS_FPCA = 100
PARAMS = {"regu": REGU, "kin_sigma": KIN_SIGMA, "kout_sigma": KOUT_SIGMA, "keval_sigma": KEVAL_SIGMA,
          "n_fpca": N_FPCA, "n_evals_fpca": N_EVALS_FPCA, "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
          "domain_in": DOMAIN, "domain_out": DOMAIN}

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

        Ytest = disc_fd1.to_discrete_general(*Ytest)

        # ############################# Full cross-validation experiment ###############################################
        # Generate configurations and regressors
        configs, regs = generate_expes.dti_kam(**PARAMS)
        # Run tuning in parallel
        result = perf_timing.parallel_perf_counter(regs, Xtrain, Ytrain, Xtest, Ytest, n_procs=N_PROCS,
                                                   min_nprocs=MIN_PROCS)
        results.append(result)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
    print(results)
