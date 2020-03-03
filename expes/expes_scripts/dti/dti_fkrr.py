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
from data.DEPRECATED import processing
from expes.DEPRECATED.expes_scripts.dti import config as config

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_fkrr"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_fkrr"
# Exec config
NPROCS = 8

# ############################### Regressor config #####################################################################
# Domain
DOMAIN_OUT = np.array([[0, 1]])
# Pre cross validated dict
CV_DICT = {'ker_sigma': 0.9, 'center_outputs': True, 'regu': 0.00271858824273294, 'ky': 0.1}
# Regularization parameter grid
REGU_GRID = np.geomspace(1e-6, 1e-1, 100)
# Output kernel bandwidth grid
KY_GRID = [0.01, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
# Input kernel standard deviation
KER_SIGMA = 0.9
# Location used for discrete approximation
APPROX_LOCS = np.linspace(0, 1, 200)

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
    cca, rcst = loading.load_dti(DATA_PATH, shuffle_seed=config.SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca.copy(), rcst.copy(),
                                                                  n_train=config.N_TRAIN, normalize01=True)
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[1]
    except IndexError:
        argv = ""
    if argv == "full":
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "ky": KY_GRID, "center_outputs": True}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_fkr_dti(expdict, APPROX_LOCS) for expdict in expe_dicts]
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
        best_regressor = generate_expes.create_fkr_dti(CV_DICT, APPROX_LOCS)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        preds = best_regressor.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))