import numpy as np
import os
import sys
import pickle
import pathlib
from time import perf_counter

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)

# Local imports
from expes import generate_expes
from model_eval import parallel_tuning
from functional_regressors import kernels
from functional_regressors import regularization
from model_eval import expe_generation
from data import loading
from functional_regressors import kernel_projection_learning as kproj

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_kpl"
# Number of processors
NPROCS = 8

# ############################### Regressor config #####################################################################
# Output domain
DOMAIN_OUT = np.array([[0, 1]])
# Padding parameters
PAD_WIDTH = ((0, 0), (0, 0))
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICTS = dict()
CV_DICTS["LP"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                  'penalize_eigvals': 0, 'n_fpca': 30, 'penalize_pow': 1}
CV_DICTS["LA"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                  'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["TBCL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                    'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["TBCD"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                    'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["VEL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                   'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["GLO"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                   'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["TTCL"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                    'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
CV_DICTS["TTCD"] = {'ker_sigma': 1, 'center_output': True, 'regu': 1e-10,
                    'penalize_eigvals': 0, 'n_fpca': 40, 'penalize_pow': 1}
# Regularization parameters grid
# REGU_GRID = list(np.geomspace(1e-10, 1e-5, 40))
REGU_GRID = [1e-10, 1e-7]
# Number of principal components to consider
N_FPCA = [20, 30]
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
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
        output_basis_configs = expe_generation.configs_combinations(
            {"n_basis": N_FPCA, "input_dim": 1, "domain": DOMAIN_OUT, "n_evals": NEVALS_FPCA},
            exclude_list={"domain"})
        output_bases = [("functional_pca", config) for config in output_basis_configs]
        ker_sigmas = np.ones(13)
        gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
        multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
        # Generate config dictionaries
        B = regularization.Eye()
        params = {"regu": REGU_GRID, "kernel_scalar": multi_ker, "output_basis": output_bases, "B": B}
        expe_dicts = expe_generation.configs_combinations(params)
        for di in expe_dicts:
            print(di)
        # Create a queue of regressor to cross validate
        regressors = [kproj.SeperableKPL(**config) for config in expe_dicts]
        # Cross validation of the regressor queue
        best_dict, best_result, score_test = parallel_tuning.parallel_tuning(regressors, Xtrain, Ytrain, Xtest, Ytest,
                                                                             rec_path, key, expe_dicts, n_procs=4)
        # Save the results
        with open(rec_path + "/" + EXPE_NAME + "_" + key + ".pkl", "wb") as inp:
            pickle.dump((best_dict, best_result, score_test), inp,
                        pickle.HIGHEST_PROTOCOL)
        # Print the result
        print("Score on test set: " + str(score_test))

    # ############################# Reduced experiment with the pre cross validated configuration ######################
    else:
        # Use directly the regressor stemming from the cross validation
        best_regressor = generate_expes.create_kpl_speech(CV_DICTS[key], NEVALS_FPCA)
        start = perf_counter()
        best_regressor.fit(Xtrain, Ytrain)
        end = perf_counter()
        print(end - start)
        # Evaluate it on test set
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
        score_test = model_eval.mean_squared_error(preds, Ytest[1])
        # Print the result
        print("Score on test set: " + str(score_test))