import numpy as np
import os
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent)
sys.path.append(path)

# Local imports
from expes.DEPRECATED import generate_expes
from model_eval import parallel_tuning
from data import loading
from data import processing
from expes import generate_expes

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kpl"
# Number of processors
NPROCS = 8
SHUFFLE_SEED = 784
N_TRAIN = 70

# ############################### Regressor config #####################################################################
# Signal extension method
SIGNAL_EXT = ("symmetric", (1, 1))
CENTER_OUTPUT = True
DOMAIN_OUT = np.array([[0, 1]])
LOCS_BOUNDS = np.array([[0 - SIGNAL_EXT[1][0], 1 + SIGNAL_EXT[1][1]]])
DECREASE_BASE = [1, 1.2]
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
PARAMS_DICT = {'ker_sigma': 0.9, 'center_outputs': True, 'regu': 0.009236708571873866}
MOMENTS = [2, 3]
BASIS_DICT = {"pywt_name": "db", "moments": MOMENTS, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
              "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}
# Standard deviation parameter for the input kernel
KER_SIGMA = 0.9
# Regularization grid
REGUS = [1e-4, 1e-3]

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
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)

    # Put input data in array form
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    #
    configs, regs = generate_expes.dti_wavs_kpl(KER_SIGMA, REGUS,
                                                center_output=CENTER_OUTPUT,
                                                signal_ext=SIGNAL_EXT,
                                                decrease_base=DECREASE_BASE,
                                                **BASIS_DICT)

    best_config, best_result, score_test = parallel_tuning.parallel_tuning(regs, Xtrain, Ytrain, Xtest, Ytest,
                                                                           rec_path=rec_path, configs=configs,
                    cv_mode="vector", n_folds=5, n_procs=None, min_nprocs=4, timeout_sleep=3,
                    n_timeout=0, cpu_avail_thresh=30)
    # Evaluate it on test set
    preds = best_regressor.predict_evaluate_diff_locs(Xtest, Ytest[0])
    score_test = model_eval.mean_squared_error(preds, Ytest[1])
    # Print the result
    print("Score on test set: " + str(score_test))