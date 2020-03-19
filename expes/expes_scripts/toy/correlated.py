import numpy as np
import os
import sys
import pathlib

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent.parent.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from data import toy_data_spline
from data import degradation
from functional_regressors import kernels
from model_eval import metrics
from data import loading, processing
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors import triple_basis
from model_eval import parallel_tuning
from functional_data import basis

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "toy"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "correlated"
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
N_PROCS = 8

# ############################### Regressor config #####################################################################
# REGU = np.geomspace(1e-7, 1e-1, 100)
REGU = [1e-4, 1e-3]
N_SAMPLES = [10, 25, 50, 100, 200]
KER_SIGMA = 20

NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02
SEED_INPUT = 768
SEED_OUTPUT = 456

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
    configs_corr, regs_corr = generate_expes.toy_spline_kpl_corr(KER_SIGMA, REGU)
    configs, regs = generate_expes.toy_spline_kpl(KER_SIGMA, REGU)

    scores_test_corr = []
    scores_test = []

    for n_samples in N_SAMPLES:
        Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data_correlated(n_samples)
        Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, SEED_INPUT)
        Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, SEED_OUTPUT)
        best_config_corr, best_result_corr, score_test_corr = parallel_tuning.parallel_tuning(
            regs_corr, Xtrain_deg, Ytrain_deg, Xtest, Ytest, rec_path=rec_path,
            configs=configs_corr, n_folds=N_FOLDS, n_procs=N_PROCS,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, rec_path=rec_path,
            configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
        scores_test_corr.append(score_test_corr)
        scores_test.append(score_test)
        print(n_samples)




