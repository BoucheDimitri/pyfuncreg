import numpy as np
import os
import pickle
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from data import loading
from expes import generate_expes
from data import processing
from model_eval import parallel_tuning, metrics

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_3be"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_3be"
# Number of processors
N_PROCS = 8
N_FOLDS = 5
INPUT_INDEXING = "list"
OUTPUT_INDEXING = "discrete_general"

# ############################### Regressor config #####################################################################
SIGNAL_EXT = ("symmetric", (0, 0))
# Output domain
DOMAIN = np.array([[0, 1]])
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICTS = dict()
CV_DICTS["LP"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 30}
CV_DICTS["LA"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["TBCL"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["TBCD"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["VEL"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["GLO"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["TTCL"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
CV_DICTS["TTCD"] = {'ker_sigma': 1, 'regu': 1e-10, 'n_fpca': 40}
# Regularization parameters grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA = 40
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300

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
    X, Y = loading.load_raw_speech_dataset(path + "/data/dataspeech/raw/")
    Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
        X, Y, shuffle_seed=784, n_train=300, normalize_domain=True, normalize_values=True)

    try:
        key = sys.argv[1]
    except IndexError:
        raise IndexError(
            'You need to define a vocal tract subproblem in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}')
    # key = "LA"
    Ytrain_ext, Ytrain, Ytest_ext, Ytest = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[2]
    except IndexError:
        argv = ""
    # argv = "full"
    if argv == "full":

        # Generate configs and corresponding regressors
        configs, regs = generate_expes.speech_fpca_3be(KER_SIGMA, REGU_GRID, N_FPCA, NEVALS_FPCA, DOMAIN)

        # Cross validation of the regressors
        best_dict, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain_ext, Xtest, Ytest, Xpred_train=None, Ypred_train=Ytrain,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            rec_path=rec_path, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + "_" + key + ".pkl", "wb") as inp:
            pickle.dump((best_dict, best_result, score_test), inp, pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        configs, regs = generate_expes.speech_fpca_3be(**CV_DICTS[key], n_evals_fpca=NEVALS_FPCA, domain=DOMAIN)
        regs[0].fit(Xtrain, Ytrain_ext)
        # Evaluate it on test set
        preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(Ytest[1], preds)
        # Print the result
        print("Score on test set: " + str(score_test))