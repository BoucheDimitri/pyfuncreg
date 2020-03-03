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

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_ke"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_ke"
# Exec config
NPROCS = 8

# ############################### Regressor config #####################################################################
CV_DICT = {'window': 0.2773737373737374}
WINDOW_GRID = [np.sqrt(t) for t in np.linspace(0.02, 1, 100)]


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
        params = {"window": WINDOW_GRID}

        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_ke_dti(expdict)
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
        best_regressor = generate_expes.create_ke_dti(CV_DICT)
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))