import numpy as np
import os
import sys
import pickle
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)

# Local imports
from expes import generate_expes
from misc import model_eval
from data import loading, processing
from expes.expes_scripts.dti import config as config

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_3be"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_3be"
# Number of processors
NPROCS = 8

# ############################### Fixed global variables ###############################################################
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICTS = dict()
CV_DICTS["LP"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 30}
CV_DICTS["LA"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["TBCL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["TBCD"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["VEL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["GLO"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["TTCL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
CV_DICTS["TTCD"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10, 'nfpca': 40}
# Output domain
DOMAIN_OUT = np.array([[0, 1]])
# Padding of the output
PAD_WIDTH = ((0, 0), (0, 0))
# Regularization grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-6, 1e-5]
# Standard deviation grid for input kernel
KER_SIGMA = 1
# Number of principal components used
# N_FPCA = [20, 30, 40]
N_FPCA = [20]
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
    # X, Y = loading.load_speech_dataset_bis(DATA_PATH)
    # Xtrain, Ytrain_full, Xtest, Ytest_full = processing.process_speech_dataset(
    #     X, Y, shuffle_seed=config.SHUFFLE_SEED, n_train=config.N_TRAIN, normalize01_domain=True, normalize_range=True)
    Xtrain, Ytrain_full, Xtest, Ytest_full = processing.load_processed_speech_dataset(DATA_PATH)
    try:
        key = sys.argv[1]
    except IndexError:
        raise IndexError("You need to define a vocal tract subproblem")
    Ytrain, Ytest = Ytrain_full[key], Ytest_full[key]

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[2]
    except IndexError:
        argv = ""
    if argv == "full":
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "nfpca": N_FPCA, "center_output": True}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_3be_speech(expdict, NEVALS_FPCA) for expdict in expe_dicts]
        # Cross validation of the regressor queue
        expe_dicts, results, best_ind, best_dict, best_result, score_test \
            = model_eval.exec_regressors_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
                                               rec_path=rec_path, nprocs=NPROCS)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + ".pkl", "wb") as inp:
            pickle.dump((best_dict, best_result, score_test), inp,
                        pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        best_regressor = generate_expes.create_3be_speech(CV_DICTS[key], NEVALS_FPCA)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))