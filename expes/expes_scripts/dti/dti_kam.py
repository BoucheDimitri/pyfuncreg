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
from data import loading, processing
from expes.DEPRECATED.expes_scripts.dti import config as config

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kam"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kam"
# Exec config
NPROCS = 8

# ############################### Regressor config #####################################################################
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICT = {'regu': 0.007564633275546291, 'kx': 0.1, 'ky': 0.05, 'keval': 0.1, 'nfpca': 30}
REGU_GRID = np.geomspace(1e-8, 1, 100)
KX_GRID = [0.01, 0.025, 0.05, 0.1]
KY_GRID = [0.01, 0.025, 0.05, 0.1]
KEV_GRID = [0.03, 0.06, 0.1]
NFPCA_GRID = [10, 15, 20, 30]


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

    # ############################# Full cross-validation experiment ###################################################
    try:
        argv = sys.argv[1]
    except IndexError:
        argv = ""
    if argv == "full":
        # Generate config dictionaries
        params = {"regu": REGU_GRID, "kx": KX_GRID, "ky": KY_GRID, "keval": KEV_GRID, "nfpca": NFPCA_GRID}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_kam_dti(expdict, NEVALS_IN, NEVALS_OUT, DOMAIN_OUT, DOMAIN_OUT, NEVALS_FPCA)
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
        best_regressor = generate_expes.create_kam_dti(CV_DICT, NEVALS_IN, NEVALS_OUT,
                                                       DOMAIN_OUT, DOMAIN_OUT, NEVALS_FPCA)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        len_test = len(Xtest[0])
        preds = [best_regressor.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
        # Print the result
        print("Score on test set: " + str(score_test))