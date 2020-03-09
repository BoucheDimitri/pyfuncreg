import numpy as np
import os
import sys
import pathlib

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from model_eval import parallel_tuning
from model_eval import metrics
from data import loading
from data import processing
from functional_regressors import kernels
from expes import generate_expes
from functional_data.DEPRECATED import discrete_functional_data as disc_fd
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors import kernel_projection_learning as kproj_learning
from model_eval import cross_validation

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kpl"
# Number of processors
N_PROCS = 8
SHUFFLE_SEED = 784
N_TRAIN = 70
N_FOLDS = 5
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"

# ############################### Regressor config #####################################################################
# Signal extension method
SIGNAL_EXT = ("symmetric", (1, 1))
CENTER_OUTPUT = True
DOMAIN_OUT = np.array([[0, 1]])
LOCS_BOUNDS = np.array([[0 - SIGNAL_EXT[1][0], 1 + SIGNAL_EXT[1][1]]])
DECREASE_BASE = [1, 1.2]
MOMENTS = [2]
BASIS_DICT = {"pywt_name": "db", "moments": MOMENTS, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
              "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}
# Standard deviation parameter for the input kernel
KER_SIGMA = 0.9
# Regularization grid
REGUS = [1e-4, 1e-3]

# ############################### Pre cross-validated config ###########################################################
# TODO: POUR LA VERSION EXACTE CE N EST PAS LA MIEUX, IL FAUDRAIT REFAIRE LA CROSS VAL
BASIS_DICT_CV = {"pywt_name": "db", "moments": 2, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
                 "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}

PARAMS_DICT_CV = {'ker_sigma': 0.9, 'center_output': True, 'regu': 0.009236708571873866, "decrease_base": 1.2}


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

    # Define execution mode
    # try:
    #     argv = sys.argv[1]
    # except IndexError:
    #     argv = ""
    argv = "full"
    # ############################# Load the data ######################################################################
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)
    # Extend data
    Ytrain_extended = disc_fd1.extend_signal_samelocs(Ytrain[0][0], Ytrain[1], mode=SIGNAL_EXT[0], repeats=SIGNAL_EXT[1])
    # Convert testing output data to discrete general form
    Ytest = disc_fd1.to_discrete_general(*Ytest)

    # Put input data in array form
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    # ############################# Full cross-validation experiment ###################################################
    if argv == "full":
        # Generate configurations and regressors
        configs, regs = generate_expes.dti_wavs_kpl(KER_SIGMA, REGUS, center_output=CENTER_OUTPUT,
                                                    decrease_base=DECREASE_BASE, **BASIS_DICT)

        best_config, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain_extended, Xtest, Ytest, None, Ytrain,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            rec_path=rec_path, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        print("Score on test set: " + str(score_test))

    else:
        # Generate regressor from cross-validation dictionaries
        configs, regs = generate_expes.dti_wavs_kpl(**PARAMS_DICT_CV, **BASIS_DICT_CV)
        regs[0].fit(Xtrain, Ytrain_extended)
        preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(Ytest[1], preds)
        print("Score on test set: " + str(score_test))

