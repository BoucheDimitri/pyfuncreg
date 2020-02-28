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

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_ke"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_ke"
# Exec config
NPROCS = 8

# ############################### Regressor config #####################################################################
# Pre cross validated dictionaries
CV_DICTS = dict()
CV_DICTS["LA"] = {'center_output': False, 'ker_sigma': 0.3}
CV_DICTS["TBCL"] = {'center_output': False, 'ker_sigma': 0.3}
CV_DICTS["TBCD"] = {'center_output': False, 'ker_sigma': 0.3}
CV_DICTS["VEL"] = {'center_output': False, 'ker_sigma': 0.2}
CV_DICTS["GLO"] = {'center_output': False, 'ker_sigma': 0.2}
CV_DICTS["TTCL"] = {'center_output': False, 'ker_sigma': 0.2}
CV_DICTS["TTCD"] = {'center_output': False, 'ker_sigma': 0.3}
# Kernel standard deviation
# KER_SIGMA = np.arange(0.1, 2.1, 0.1)
KER_SIGMA = [0.1, 1]


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
        params = {"ker_sigma": KER_SIGMA, "center_output": False}
        expe_dicts = generate_expes.expe_generator(params)
        # Create a queue of regressor to cross validate
        regressors = [generate_expes.create_ke_speech(expdict) for expdict in expe_dicts]
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
        best_regressor = generate_expes.create_ke_speech(CV_DICTS[key])
        best_regressor.fit(Xtrain, Ytrain)
        # Evaluate it on test set
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))