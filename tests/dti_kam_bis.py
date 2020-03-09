import numpy as np
import os
import sys
import pathlib
import importlib

# # Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# # path = os.getcwd()
# sys.path.append(path)
path = os.getcwd()

# Local imports
from model_eval import parallel_tuning
from model_eval import metrics
from model_eval import cross_validation
from data import loading
from data import processing
from expes import generate_expes
from functional_data.DEPRECATED import discrete_functional_data as disc_fd
from functional_regressors import kernels
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors import kernel_additive

importlib.reload(cross_validation)
importlib.reload(parallel_tuning)

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kam"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kam"
# Number of processors
N_PROCS = 8
SHUFFLE_SEED = 784
N_TRAIN = 70
N_FOLDS = 5
INPUT_DATA_FORMAT = "discrete_general"
OUTPUT_DATA_FORMAT = 'discrete_general'

# ############################### Regressor config #####################################################################
# REGU_GRID = np.geomspace(1e-8, 1, 100)
# KX_GRID = [0.01, 0.025, 0.05, 0.1]
# KY_GRID = [0.01, 0.025, 0.05, 0.1]
# KEV_GRID = [0.03, 0.06, 0.1]
# NFPCA_GRID = [10, 15, 20, 30]
REGU_GRID = [1e-4, 1e-5]
KX_GRID = 0.01
KY_GRID = 0.01
KEV_GRID = 0.03
NFPCA_GRID = [10, 15]
DOMAIN = np.array([[0, 1]])
N_EVALS_IN = 100
N_EVALS_OUT = 100
N_EVALS_FPCA = 100
PARAMS = {"regu": REGU_GRID, "kx_sigma": KX_GRID, "ky_sigma": KY_GRID, "keval_sigma": KEV_GRID,
          "n_fpca": NFPCA_GRID, "n_evals_fpca": N_EVALS_FPCA, "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
          "domain_in": DOMAIN, "domain_out": DOMAIN}

# ############################### Pre cross-validated config ###########################################################
PARAMS_CV = {"regu": 0.007564633275546291, "kx_sigma": 0.1, "ky_sigma": 0.05, "keval_sigma": 0.1,
             "n_fpca": 30, "n_evals_fpca": N_EVALS_FPCA, "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
             "domain_in": DOMAIN, "domain_out": DOMAIN}

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
    #
    # # Define execution mode
    # try:
    #     argv = sys.argv[1]
    # except IndexError:
    #     argv = ""

    # ############################# Load the data ######################################################################
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)

    # Put data in discrete general form
    Xtrain, Xtest = disc_fd1.set_locs(*Xtrain), disc_fd1.set_locs(*Xtest)
    Ytrain, Ytest = disc_fd1.set_locs(*Ytrain), disc_fd1.set_locs(*Ytest)

    Ytest = disc_fd1.to_discrete_general(*Ytest)

    kx = kernels.GaussianScalarKernel(KX_GRID, normalize=False)
    ky = kernels.GaussianScalarKernel(KY_GRID, normalize=False)
    keval = kernels.GaussianScalarKernel(KEV_GRID, normalize=False)

    params = {"regu": 1e-4, "kerlocs_in": kx, "kerlocs_out": ky, "kerevals": keval,
              "n_fpca": 20, "n_evals_fpca": N_EVALS_FPCA,
              "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
              "domain_in": DOMAIN, "domain_out": DOMAIN}

    test_kam = kernel_additive.KernelAdditiveModelBis(**params)

    test_kam.fit(Xtrain, Ytrain)

    preds = test_kam.predict_evaluate_diff_locs(Xtest, Ytest[0])
    score_test = metrics.mse(Ytest[1], preds)

    # ############################# Full cross-validation experiment ###################################################
    # if argv == "full":
    #     # Generate configurations and regressors
    #     configs, regs = generate_expes.dti_kam(**PARAMS)
    #     # Run tuning in parallel
    #     best_config, best_result, score_test = parallel_tuning.parallel_tuning(
    #         regs, Xtrain, Ytrain, Xtest, Ytest, rec_path=rec_path, configs=configs, input_data_format=INPUT_DATA_FORMAT,
    #         output_data_format=OUTPUT_DATA_FORMAT, n_folds=N_FOLDS, n_procs=N_PROCS)
    #     print("Score on test set: " + str(score_test))
    #
    # # ############################# Use pre cross-validated dictionary #################################################
    # else:
    #     # Generate regressor from cross-validation dictionaries
    #     configs, regs = generate_expes.dti_kam(**PARAMS_CV)
    #     regs[0].fit(Xtrain, Ytrain, output_data_format=OUTPUT_DATA_FORMAT)
    #     Ytest_dg = disc_fd.to_discrete_general(Ytest, OUTPUT_DATA_FORMAT)
    #     preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest_dg[0])
    #     score_test = metrics.mse(Ytest_dg[1], preds)
    #     print("Score on test set: " + str(score_test))
    #
