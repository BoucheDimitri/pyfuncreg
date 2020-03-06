import numpy as np
import os

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from model_eval import metrics
from data import loading
from functional_regressors import kernels
from functional_regressors import triple_basis
from data import processing
from functional_data.DEPRECATED import discrete_functional_data as disc_fd

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/processed/"
# Record config
OUTPUT_FOLDER = "speech_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "speech_kpl"
# Number of processors
NPROCS = 8
INPUT_DATA_FORMAT = "vector"
OUTPUT_DATA_FORMAT = "discrete_samelocs_regular_1d"

# ############################### Regressor config #####################################################################
SIGNAL_EXT = ("symmetric", (0, 0))
# Output domain
DOMAIN = np.array([[0, 1]])
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
N_FPCA = 40
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Number of evaluations for FPCA
NEVALS_FPCA = 300



# ############################## Pre cross-validated dict ##############################################################


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
    Xtrain, Ytrain_full, Xtest, Ytest_full = loading.load_speech_dataset(DATA_PATH)
    key = "LA"
    Xtrain, Ytrain, Xtest, Ytest = processing.process_speech_1VT(Xtrain, Ytrain_full, Xtest, Ytest_full, key)
    # try:
    #     key = sys.argv[1]
    # except IndexError:
    #     raise IndexError(
    #         'You need to define a vocal tract subproblem in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}')
    # ############################# Full cross-validation experiment ###################################################
    # try:
    #     argv = sys.argv[2]
    # except IndexError:
    #     argv = ""
    # if argv == "full":
    #     # Generate config dictionaries
    #     params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "penalize_eigvals": 0, "n_fpca": N_FPCA, "center_output": True}

    output_basis = ("functional_pca", {"n_basis": N_FPCA, "input_dim": 1, "domain": DOMAIN, "n_evals": NEVALS_FPCA})
    ker_sigmas = np.ones(13)
    gauss_kers = [kernels.GaussianScalarKernel(sig, normalize=False, normalize_dist=True) for sig in ker_sigmas]
    multi_ker = kernels.SumOfScalarKernel(gauss_kers, normalize=False)
    regu = 1e-10
    bibasis_test = triple_basis.BiBasisEstimator(multi_ker, output_basis, regu, center_output=True, signal_ext=SIGNAL_EXT)
    bibasis_test.fit(Xtrain, Ytrain, input_data_format=INPUT_DATA_FORMAT, output_data_format=OUTPUT_DATA_FORMAT)

    Ytest_dg = disc_fd.to_discrete_general(Ytest, OUTPUT_DATA_FORMAT)
    preds = bibasis_test.predict_evaluate_diff_locs(Xtest, Ytest_dg[0])
    score_test = metrics.mse(Ytest_dg[1], preds)
    print("Score on test set: " + str(score_test))