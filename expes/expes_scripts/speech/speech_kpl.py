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

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kpl"
# Number of processors
NPROCS = 8

# ############################### Regressor config #####################################################################
# Output domain
DOMAIN_OUT = np.array([[0, 1]])
# Padding parameters
PAD_WIDTH_OUTPUT = ((0, 0), (3*55, 3*55))
PAD_WIDTH_INPUT = ((0, 0), (0, 0))
DOMAIN_OUT_PAD = np.array([[-PAD_WIDTH_OUTPUT[1][0] / 55, 1 + PAD_WIDTH_OUTPUT[1][0] / 55]])
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICT = {'ker_sigma': 0.9,'pywt_name': 'db', 'init_dilat': 1, 'dilat': 2, 'translat': 1,
           'n_dilat': 5, 'center_outputs': True, 'add_constant': True, 'regu': 0.009236708571873866,
           'moments': 2, 'penalize_freqs': 1.2}
# Regularization parameters grid
REGU_GRID = np.geomspace(1e-8, 1, 100)
# Wavelet name for the dictionary
PYWT_NAME = "db"
# Number of vanishing moments to test
MOMENTS = (2, 3)
# Number of dilations to test
NDILATS = (4, 5)
# Bases for penalization of smaller scales
FREQS_PEN = (1.0, 1.2, 1.4, 1.6)
# Standard deviation parameter for the input kernel
KER_SIGMA = 0.9

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
                                                                  n_train=config.N_TRAIN, normalize01=True,
                                                                  pad_width_output=PAD_WIDTH_OUTPUT)
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[1]
    except IndexError:
        argv = ""
    if argv == "full":
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "pywt_name": PYWT_NAME,
                  "init_dilat": 1, "dilat": 2, "translat": 1,
                  "moments": MOMENTS, "n_dilat": NDILATS, "center_outputs": True,
                  "penalize_freqs": FREQS_PEN, "add_constant": True}

        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_kpl_dti(expdict, DOMAIN_OUT, DOMAIN_OUT_PAD, PAD_WIDTH_OUTPUT)
                      for expdict in expe_dicts]
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
        best_regressor = generate_expes.create_kpl_dti(CV_DICT, DOMAIN_OUT, DOMAIN_OUT_PAD, PAD_WIDTH_OUTPUT)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        preds = best_regressor.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))