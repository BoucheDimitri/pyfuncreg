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
from model_eval import parallel_tuning
from data import loading, processing
from functional_data import discrete_functional_data as disc_fd

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_ke_multi"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_ke"

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
KX_SIGMA = np.linspace(0.05, 2, 200)
# KX_SIGMA = [0.1, 0.5, 1]

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
        # Convert testing output data to discrete general form
        Ytest = disc_fd.to_discrete_general(*Ytest)
        # Put input data in array form
        Xtrain = np.array(Xtrain[1]).squeeze()
        Xtest = np.array(Xtest[1]).squeeze()
        configs, regs = generate_expes.dti_ke(KX_SIGMA)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain, Xtest, Ytest, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS, min_nprocs=MIN_PROCS)
        best_configs.append(best_config)
        best_results.append(best_results)
        scores_test.append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((best_configs, best_results, scores_test), out, pickle.HIGHEST_PROTOCOL)