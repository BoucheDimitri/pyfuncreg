import numpy as np
import os
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from expes import generate_expes
from model_eval import metrics, parallel_tuning
from data import loading, processing
from functional_data import discrete_functional_data as disc_fd

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_fkrr"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_fkrr"
# Exec config
N_PROCS = 8
N_FOLDS = 5
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"

# ############################### Regressor config #####################################################################
# Domain
DOMAIN_OUT = np.array([[0, 1]])
# Pre cross validated dict
CV_DICT = {'kx_sigma': 0.9, 'center_output': True, 'regu': 0.00271858824273294, 'ky_sigma': 0.1}
# Regularization parameter grid
# REGU_GRID = np.geomspace(1e-6, 1e-1, 100)
# REGU_GRID = 0.00271858824273294
REGU_GRID = [1e-4, 1e-3]
# Output kernel bandwidth grid
# KY_GRID = [0.01, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
KY_GRID = [0.01, 0.1]
# Input kernel standard deviation
KER_SIGMA = 0.9
# Location used for discrete approximation
APPROX_LOCS = np.linspace(0, 1, 200)
#
CENTER_OUTPUT = True

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
    # Convert testing output data to discrete general form
    Ytest = disc_fd.to_discrete_general(*Ytest)
    # Put input data in array form
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    # ############################# Full cross-validation experiment ###################################################
    if argv == "full":
        configs, regs = generate_expes.dti_fkrr(KER_SIGMA, KY_GRID, REGU_GRID, APPROX_LOCS, CENTER_OUTPUT)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain, Xtest, Ytest, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            rec_path=rec_path, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        configs, regs = generate_expes.dti_fkrr(**CV_DICT, approx_locs=APPROX_LOCS)
        regs[0].fit(Xtrain, Ytrain)
        preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(Ytest[1], preds)
        print("Score on test set: " + str(score_test))
