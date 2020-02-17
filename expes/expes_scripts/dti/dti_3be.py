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
DATA_PATH = path + "/data/dataDTI/"
# Method config
DOMAIN_OUT = np.array([[0, 1]])
PAD_WIDTH = ((0, 0), (0, 0))
N_RFFS = 300
RFFS_SEED = 567
# Record config
OUTPUT_FOLDER = "dti_3be"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_3be"
# Exec config
NPROCS = 8

# ############################### Fixed global variables ###############################################################
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICT = {'center_outputs': True, 'regu': 1.0, 'ker_sigma': 20, 'max_freq_in': 25, 'max_freq_out': 5}
REGU_GRID = list(np.geomspace(1e-8, 1, 100))
KER_SIGMA = [20, 25, 30, 35, 40]
FREQS_IN_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
FREQS_OUT_GRID = [5, 10, 15, 20, 25, 30, 35, 40]


if __name__ == '__main__':

    # ############################# GLOBAL PARAMETERS ##################################################################
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

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[1]
    except IndexError:
        argv = ""
    if argv == "full":
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "max_freq_in": FREQS_IN_GRID,
                  "max_freq_out": FREQS_OUT_GRID, "center_outputs": True}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_3be_dti(expdict, DOMAIN_OUT, DOMAIN_OUT, N_RFFS, RFFS_SEED, PAD_WIDTH)
                      for expdict in expe_dicts]
        # Cross validation of the regressor queue
        expe_dicts, results, best_ind, best_dict, best_result, score_test \
            = model_eval.exec_regressors_eval_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
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
        best_regressor = generate_expes.create_3be_dti(CV_DICT, DOMAIN_OUT, DOMAIN_OUT, N_RFFS, RFFS_SEED, PAD_WIDTH)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        len_test = len(Xtest[0])
        preds = [best_regressor.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
        # Print the result
        print("Score on test set: " + str(score_test))