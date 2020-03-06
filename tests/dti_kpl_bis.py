import numpy as np
import os
import sys
import pathlib

# Execution path
# exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# path = str(exec_path.parent)
# sys.path.append(path)
path = os.getcwd()

# Local imports
from model_eval import parallel_tuning
from model_eval import metrics
from data import loading
from data import processing
from functional_regressors import kernels
from expes import generate_expes
from functional_data.DEPRECATED import discrete_functional_data as disc_fd
from functional_data import discrete_functional_data as disc_fd1
from functional_regressors import kernel_projection_learning as kproj_learning

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataDTI/"
# Record config
OUTPUT_FOLDER = "dti_kpl"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "dti_kpl"
# Number of processors
N_PROCS = 8
SHUFFLE_SEED = 784
N_TRAIN = 70
N_FOLDS = 5
INPUT_DATA_FORMAT = "vector"
OUTPUT_DATA_FORMAT = 'discrete_samelocs_regular_1d'

# ############################### Regressor config #####################################################################
# Signal extension method
SIGNAL_EXT = ("symmetric", (1, 1))
CENTER_OUTPUT = True
DOMAIN_OUT = np.array([[0, 1]])
LOCS_BOUNDS = np.array([[0 - SIGNAL_EXT[1][0], 1 + SIGNAL_EXT[1][1]]])
DECREASE_BASE = [1, 1.2]
MOMENTS = [2]
BASIS_DICT = {"pywt_name": "db", "moments": MOMENTS, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
              "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}
# Standard deviation parameter for the input kernel
KER_SIGMA = 0.9
# Regularization grid
REGUS = [1e-4, 1e-3]

# ############################### Pre cross-validated config ###########################################################
PARAMS_DICT_CV = {'ker_sigma': 0.9, 'center_output': True, 'regu': 0.009236708571873866, "decrease_base": 1.2}
BASIS_DICT_CV = {"pywt_name": "db", "moments": 2, "init_dilat": 1.0, "translat": 1.0, "dilat": 2, "approx_level": 6,
                 "add_constant": True, "domain": DOMAIN_OUT, "locs_bounds": LOCS_BOUNDS}


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

    # Define execution mode
    # try:
    #     argv = sys.argv[1]
    # except IndexError:
    #     argv = ""
    argv = "full"
    # ############################# Load the data ######################################################################
    cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)
    Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)
    Ytrain_extended = disc_fd1.extend_signal_samelocs(Ytrain[0], Ytrain[1], mode="symmetric", repeats=(1, 1))

    # Put data in discrete general form
    Ytrain_extended = disc_fd1.set_locs(Ytrain_extended[0], Ytrain_extended[1])
    Ytest = disc_fd1.set_locs(Ytest[0], Ytest[1])
    Ytest = disc_fd1.to_discrete_general(Ytest[0], Ytest[1])

    # Put input data in array form
    Xtrain = np.array(Xtrain[1]).squeeze()
    Xtest = np.array(Xtest[1]).squeeze()

    # ############################# Full cross-validation experiment ###################################################
    if argv == "full":
        # Generate configurations and regressors
        # configs, regs = generate_expes.dti_wavs_kpl(KER_SIGMA, REGUS, center_output=CENTER_OUTPUT,
        #                                             signal_ext=SIGNAL_EXT, decrease_base=DECREASE_BASE, **BASIS_DICT)

        basis = ("wavelets", BASIS_DICT_CV)
        output_matrix = ("wavelets_pow", {"decrease_base": 1.2})
        ker = kernels.GaussianScalarKernel(KER_SIGMA, normalize=False)
        params = {"kernel_scalar": ker, "B": output_matrix, "output_basis": basis,
                  "regu": 0.009236708571873866, "center_output": True, "signal_ext": None}
        reg = kproj_learning.SeperableKPLBis(ker, B=output_matrix, output_basis=basis, regu=0.009236708571873866, center_output=True)
        reg.fit(Xtrain, Ytrain_extended)
        preds = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
        score_test = metrics.mse(Ytest[1], preds)

        # TODO: ADAPTER TOUT EN ENLEVANT LES OUTPUT_DATA_FORMAT DE PARTOUT + DONNER LA POSSIBILITE EN CROSSVAL
        # TODO D ENTRAINER ET DE TESTER SUR DES DATASETS DIFFERENTS + DE FILER DES LISTES CUSTOMS DE
        # TODO DONNEES DENTRAINEMENT POUR LES DIFFERENTS REGRESSORS SI PAR EXEMPLES ON VEUT EN ENTRAINER
        # TODO CERTAINS SUR DES DONNEES ETENDUES ET D AUTRES NON

    #     # Run tuning in parallel
    #     best_config, best_result, score_test = parallel_tuning.parallel_tuning(
    #         regs, Xtrain, Ytrain, Xtest, Ytest, rec_path=rec_path, configs=configs, input_data_format=INPUT_DATA_FORMAT,
    #         output_data_format=OUTPUT_DATA_FORMAT, n_folds=N_FOLDS, n_procs=N_PROCS)
    #     print("Score on test set: " + str(score_test))
    #
    # else:
    #     # Generate regressor from cross-validation dictionaries
    #     configs, regs = generate_expes.dti_wavs_kpl(**PARAMS_DICT_CV, signal_ext=SIGNAL_EXT, **BASIS_DICT_CV)
    #     regs[0].fit(Xtrain, Ytrain, output_data_format=OUTPUT_DATA_FORMAT)
    #     Ytest_dg = disc_fd.to_discrete_general(Ytest, OUTPUT_DATA_FORMAT)
    #     preds = regs[0].predict_evaluate_diff_locs(Xtest, Ytest_dg[0])
    #     score_test = metrics.mse(Ytest_dg[1], preds)
    #     print("Score on test set: " + str(score_test))
    #
