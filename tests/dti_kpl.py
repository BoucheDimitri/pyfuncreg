import numpy as np
import os
import sys
import pickle
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent)
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
PARAMS_DICT = {'ker_sigma': 0.9, 'center_outputs': True, 'regu': 0.009236708571873866}
# BASIS_DICT = {"domain": DOMAIN_OUT, "locs_bounds": DOMAIN_OUT_PAD,
#               'pywt_name': 'db', 'init_dilat': 1, 'dilat': 2, 'translat': 1,
#               'n_dilat': 5, 'add_constant': True}
BASIS_DICT = {"domain": DOMAIN_OUT, "input_dim": 1, "n_basis": 20, "n_evals": 100}
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
    print(Xtrain[0][0].shape)
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    best_regressor = generate_expes.create_kpl_dti_bis(PARAMS_DICT, BASIS_DICT)
    best_regressor.fit(Xtrain, Ytrain)
    # Evaluate it on test set
    preds = best_regressor.predict_evaluate_diff_locs(Xtest, Ytest[0])
    score_test = model_eval.mean_squared_error(preds, Ytest[1])
    # Print the result
    print("Score on test set: " + str(score_test))