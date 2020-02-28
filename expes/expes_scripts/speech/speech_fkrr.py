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
from expes.DEPRECATED import generate_expes
from misc import model_eval
from data import loading

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_fkrr"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_fkrr"
# Exec config
NPROCS = 8

# ############################### Regressor config #####################################################################
# Domain
DOMAIN_OUT = np.array([[0, 1]])
# Pre cross validated dict
CV_DICTS = dict()
CV_DICTS["LP"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.08, 'center_outputs': True}
CV_DICTS["LA"] = {'ker_sigma': 1, 'regu': 9.426684551178854e-08, 'ky': 0.15, 'center_outputs': False}
CV_DICTS["TBCL"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.08, 'center_outputs': True}
CV_DICTS["VEL"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.15, 'center_outputs': True}
CV_DICTS["GLO"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.06, 'center_outputs': True}
CV_DICTS["TTCL"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.03, 'center_outputs': True}
CV_DICTS["TTCD"] = {'ker_sigma': 1, 'regu': 1e-10, 'ky': 0.06, 'center_outputs': True}
# Regularization parameter grid
REGU_GRID = np.geomspace(1e-10, 1e-6, 40)
# Output kernel bandwidth grid
KY_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15]
# Input kernel standard deviation
KER_SIGMA = 1
# Center outputs possibilities
CENTER_OUTPUTS = (True, False)
# Location used for discrete approximation
APPROX_LOCS = np.linspace(0, 1, 300)


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
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "ky": KY_GRID, "center_outputs": CENTER_OUTPUTS}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_fkr_speech(expdict, APPROX_LOCS) for expdict in expe_dicts]
        # Cross validation of the regressor queue
        expe_dicts, results, best_ind, best_dict, best_result, score_test \
            = model_eval.exec_regressors_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
                                               rec_path=rec_path, nprocs=NPROCS)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + "_" + key + ".pkl", "wb") as inp:
            pickle.dump((best_dict, best_result, score_test), inp,
                        pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        best_regressor = generate_expes.create_fkr_speech(CV_DICTS[key], APPROX_LOCS)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        preds = best_regressor.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))