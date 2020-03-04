import numpy as np
import os
import sys
import pathlib
import importlib

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from model_eval import parallel_tuning
from model_eval import metrics
from data import loading
from functional_regressors import triple_basis
from data import processing
from functional_data import basis
from expes import generate_expes
from functional_data import discrete_functional_data as disc_fd

importlib.reload(triple_basis)

# ############################### Config ###############################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_3be"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_3be"
# Number of processors
NPROCS = 8
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_DATA_FORMAT = 'discrete_samelocs_regular_1d'
OUTPUT_DATA_FORMAT = 'discrete_samelocs_regular_1d'

# ############################### Regressor config #####################################################################
# Dictionary obtained by cross validation for quick run fitting on train and get score on test
CV_DICT = {'center_outputs': True, 'regu': 1.0, 'ker_sigma': 20, 'max_freq_in': 25, 'max_freq_out': 5}
SIGNAL_EXT_INPUT = ("symmetric", (1, 1))
SIGNAL_EXT_OUTPUT = ("symmetric", (1, 1))
# Output domain
DOMAIN = np.array([[0, 1]])
# Padding of the output
PAD_WIDTH = ((0, 0), (0, 0))
# Number of random fourier features
N_RFFS = 300
# Seed for the random fourier features
RFFS_SEED = 567
# Regularization grid
REGU_GRID = list(np.geomspace(1e-8, 1, 100))
# Standard deviation grid for input kernel
# KER_SIGMA = [20, 25, 30, 35, 40]
KER_SIGMA = 20
# Maximum frequency to include for input and output
FREQS_IN_GRID = [5, 10, 15, 20, 25, 30, 35, 40]
FREQS_OUT_GRID = [5, 10, 15, 20, 25, 30, 35, 40]


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
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)

    MAX_FREQ_IN = 25
    MAX_FREQ_OUT = 10
    input_basis_dict = {"lower_freq": 0, "upper_freq": MAX_FREQ_IN, "domain": DOMAIN}
    basis_in = ("fourier", input_basis_dict)
    output_basis_dict = {"lower_freq": 0, "upper_freq": MAX_FREQ_OUT, "domain": DOMAIN}
    basis_out = ("fourier", output_basis_dict)
    rffs_basis_dict = {"n_basis": N_RFFS, "domain": DOMAIN, "bandwidth": KER_SIGMA, "seed": RFFS_SEED}
    rffs_basis = ("random_fourier", rffs_basis_dict)

    regu = 1

    reg = triple_basis.TripleBasisEstimator(basis_in, rffs_basis, basis_out, regu,
                                            center_output=True,
                                            signal_ext_input=SIGNAL_EXT_INPUT,
                                            signal_ext_output=SIGNAL_EXT_OUTPUT)

    reg.fit(Xtrain, Ytrain, INPUT_DATA_FORMAT, OUTPUT_DATA_FORMAT)
    Xtest = disc_fd.to_discrete_general(Xtest, INPUT_DATA_FORMAT)
    Ytest = disc_fd.to_discrete_general(Ytest, OUTPUT_DATA_FORMAT)
    preds = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
    score_test = metrics.mse(preds, Ytest[1])

    # ############################# Full cross-validation experiment ###################################################
    # try:
    #     argv = sys.argv[1]
    # except IndexError:
    #     argv = ""
    # if argv == "full":
    #     # Generate config dictionaries
    #     params = {"regu": REGU_GRID, "ker_sigma": KER_SIGMA, "max_freq_in": FREQS_IN_GRID,
    #               "max_freq_out": FREQS_OUT_GRID, "center_outputs": True}
    #     expe_dicts = generate_expes.expe_generator(params)
    #     # Create a queue of regressor to cross validate
    #     regressors = [generate_expes.create_3be_dti(expdict, DOMAIN_OUT, DOMAIN_OUT, N_RFFS, RFFS_SEED, PAD_WIDTH)
    #                   for expdict in expe_dicts]
    #     # Cross validation of the regressor queue
    #     expe_dicts, results, best_ind, best_dict, best_result, score_test \
    #         = model_eval.exec_regressors_eval_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
    #                                                 rec_path=rec_path, nprocs=NPROCS)
    #     # Save the results
    #     with open(rec_path + "/" + EXPE_NAME + ".pkl", "wb") as inp:
    #         pickle.dump((best_dict, best_result, score_test), inp,
    #                     pickle.HIGHEST_PROTOCOL)
    #     # Print the result
    #     print("Score on test set: " + str(score_test))
    #
    # # ############################# Reduced experiment with the pre cross validated configuration ######################
    # else:
    #     # Use directly the regressor stemming from the cross validation
    #     best_regressor = generate_expes.create_3be_dti(CV_DICT, DOMAIN_OUT, DOMAIN_OUT, N_RFFS, RFFS_SEED, PAD_WIDTH)
    #     best_regressor.fit(Xtrain, Ytrain)
    #     # Evaluate it on test set
    #     len_test = len(Xtest[0])
    #     preds = [best_regressor.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len_test)]
    #     score_test = model_eval.mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
    #     # Print the result
    #     print("Score on test set: " + str(score_test))