import numpy as np
import os
import sys
import pickle
import pathlib
from time import perf_counter

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)

# Local imports
from model_eval import parallel_tuning
from data import loading
from expes import generate_expes
from model_eval import metrics

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_kpl"
# Number of processors
NPROCS = 8

# ############################### Regressor config #####################################################################
# Output domain
DOMAIN_OUT = np.array([[0, 1]])
# Padding parameters
PAD_WIDTH = ((0, 0), (0, 0))
# Regularization parameters grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA_GRID = [20, 30]
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300
# Penalization grid
PENPOW_GRID = [1.0, 1.2]
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICTS = dict()
# (ker_sigma, regus, n_fpca, n_evals_fpca, decrease_pow, domain=np.array([[0, 1]]))
CV_DICTS["LP"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 30, 'n_evals_fpca': NEVALS_FPCA,
                  'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["LA"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                  'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["TBCL"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                    'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["TBCD"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                    'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["VEL"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                   'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["GLO"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                   'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["TTCL"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                    'decrease_base': 1, 'domain': DOMAIN_OUT}
CV_DICTS["TTCD"] = {'ker_sigma': 1, 'regus': 1e-10, 'n_fpca': 40, 'n_evals_fpca': NEVALS_FPCA,
                    'decrease_base': 1, 'domain': DOMAIN_OUT}

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
    Xtrain, Ytrain_full, Xtest, Ytest_full = loading.load_processed_speech_dataset(DATA_PATH)
    try:
        key = sys.argv[1]
    except IndexError:
        raise IndexError(
            'You need to define a vocal tract subproblem in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}')
    Ytrain, Ytest = Ytrain_full[key], Ytest_full[key]

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[2]
    except IndexError:
        argv = ""
    if argv == "full":
        configs, regs = generate_expes.speech_fpca_penpow_kpl(
            KER_SIGMA, REGU_GRID, N_FPCA_GRID, NEVALS_FPCA, PENPOW_GRID, DOMAIN_OUT)
        # Cross validation of the regressor queue
        best_dict, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain, Xtest, Ytest, rec_path, key, configs, n_procs=NPROCS)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + "_" + key + ".pkl", "wb") as out_file:
            pickle.dump((best_dict, best_result, score_test), out_file, pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        best_reg = generate_expes.speech_fpca_penpow_kpl(**CV_DICTS[key])
        best_reg.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        preds = best_reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))