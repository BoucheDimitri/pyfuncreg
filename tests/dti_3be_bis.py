import numpy as np
import os
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from model_eval import metrics
from data import loading
from data import processing
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors import triple_basis
from model_eval import parallel_tuning

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_3be"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_3be"
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "discrete_general"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
N_PROCS = 8

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
# REGU_GRID = list(np.geomspace(1e-8, 1, 100))
REGU_GRID = [1e-1, 1]
# Standard deviation grid for input kernel
KER_SIGMA = [20, 30, 40]
# KER_SIGMA = 20
# Maximum frequency to include for input and output
# FREQS_IN_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
# FREQS_OUT_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
FREQS_IN_GRID = [20, 25]
FREQS_OUT_GRID = [5, 10]
CENTER_OUTPUT = True

# ############################## Pre cross-validated dict ##############################################################
CV_PARAMS = {'center_output': True,
             'basis_in': ('fourier', {'lower_freq': 0, 'upper_freq': 25, 'domain': DOMAIN}),
             'basis_out': ('fourier', {'lower_freq': 0, 'upper_freq': 5, 'domain': DOMAIN}),
             'basis_rffs': ('random_fourier', {'n_basis': N_RFFS, 'domain': DOMAIN, 'seed': RFFS_SEED, 'bandwidth': 20}),
             'regu': 1}


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

    try:
        argv = sys.argv[1]
    except IndexError:
        argv = ""

    # ############################# Load the data ######################################################################
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)
    # Extend data
    Xtrain_extended = disc_fd1.extend_signal_samelocs(
        Xtrain[0][0], Xtrain[1], mode=SIGNAL_EXT_INPUT[0], repeats=SIGNAL_EXT_INPUT[1])
    Ytrain_extended = disc_fd1.extend_signal_samelocs(
        Ytrain[0][0], Ytrain[1], mode=SIGNAL_EXT_OUTPUT[0], repeats=SIGNAL_EXT_OUTPUT[1])
    # Convert testing output data to discrete general form
    Ytest = disc_fd1.to_discrete_general(*Ytest)

    # ############################# Full cross-validation experiment ###################################################
    if argv == "full":
        configs, regs = generate_expes.dti_3be_fourier(KER_SIGMA, REGU_GRID, CENTER_OUTPUT, FREQS_IN_GRID,
                                                       FREQS_OUT_GRID,N_RFFS, RFFS_SEED, DOMAIN, DOMAIN)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain_extended, Ytrain_extended, Xtest, Ytest, Xtrain, Ytrain,
            rec_path=rec_path, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        print("Score on test set: " + str(score_test))

    # ############################## Reduced experiment with the pre cross validated configuration #####################
    else:
        # Use directly the regressor stemming from the cross validation
        best_reg = triple_basis.TripleBasisEstimatorBis(**CV_PARAMS)
        best_reg.fit(Xtrain_extended, Ytrain_extended)
        preds = best_reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(preds, Ytest[1])
        print("Score on test set: " + str(score_test))
