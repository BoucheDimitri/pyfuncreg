import numpy as np
import os
import sys
import pickle
import pathlib
from time import perf_counter

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from expes import generate_expes
from data import loading, processing
from model_eval import parallel_tuning, metrics

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_fkrr"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_fkrr"
# Number of processors
N_PROCS = 8
# Indexing
INPUT_INDEXING = "list"
OUTPUT_INDEXING = "discrete_general"
# Number of folds
N_FOLDS = 5

# ############################### Regressor config #####################################################################
# Output domain
DOMAIN = np.array([[0, 1]])
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICTS = dict()
CV_DICTS["LP"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.08, 'center_output': True}
CV_DICTS["LA"] = {'kx_sigma': 1, 'regu': 9.426684551178854e-08, 'ky_sigma': 0.15, 'center_output': False}
CV_DICTS["TBCL"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.08, 'center_output': True}
CV_DICTS["VEL"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.15, 'center_output': True}
CV_DICTS["GLO"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.06, 'center_output': True}
CV_DICTS["TTCL"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.03, 'center_output': True}
CV_DICTS["TTCD"] = {'kx_sigma': 1, 'regu': 1e-10, 'ky_sigma': 0.06, 'center_output': True}
# Regularization parameters grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-10, 1e-7]
# Standard deviation parameter for the input kernel
KX_SIGMA = 1
#
KY_SIGMA = [0.02, 0.03]
# Approximation locations
APPROX_LOCS = np.linspace(0, 1, 300)
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
    # argv = "nimp"
    if argv == "full":
        # Generate configs and corresponding regressors
        configs, regs = generate_expes.speech_fkrr(KX_SIGMA, KY_SIGMA, REGU_GRID, APPROX_LOCS, CENTER_OUTPUT)

        # Cross validation of the regressors
        best_dict, best_result, score_test = parallel_tuning.parallel_tuning(
            regs, Xtrain, Ytrain_ext, Xtest, Ytest, Xpred_train=None, Ypred_train=Ytrain,
            input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING,
            rec_path=rec_path, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + "_" + key + ".pkl", "wb") as inp:
            pickle.dump((best_dict, best_result, score_test), inp,
                        pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        configs, regs = generate_expes.speech_fkrr(**CV_DICTS[key], approx_locs=APPROX_LOCS)
        regs[0].fit(Xtrain, Ytrain_ext)
        # Evaluate it on test set
        preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(Ytest[1], preds)
        # Print the result
        print("Score on test set: " + str(score_test))