import numpy as np
import os
import sys
import pickle
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
# path = os.getcwd()

# Local imports
from data import degradation
from data import toy_data_spline
from expes import generate_expes
from model_eval import parallel_tuning

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "toy_kam_kertuning"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "discrete_general"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
N_PROCS = None
MIN_PROCS = 32
# N_PROCS = 7
# MIN_PROCS = None

# ############################### Experiment parameters ################################################################
REGU = np.geomspace(1e-8, 1, 50)
KIN_SIGMA = [0.05, 0.1, 0.25, 0.5, 1, 2.5]
KOUT_SIGMA = [0.05, 0.1, 0.25, 0.5, 1, 2.5]
KEVAL_SIGMA = [0.05, 0.1, 0.25, 0.5, 1, 2.5]
# KIN_SIGMA = [0.05]
# KOUT_SIGMA = [0.01]
# KEVAL_SIGMA = [0.03]
# N_FPCA = [10, 20, 30]
# REGU = [1e-4]
# KIN_SIGMA = 0.01
# KOUT_SIGMA = 0.01
# KEVAL_SIGMA = 0.03
N_FPCA = [10, 20, 30]
NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02
NSAMPLES_LIST = [100]
MISSING_LEVELS = np.arange(0, 1, 0.05)
DOMAIN_OUT = toy_data_spline.DOM_OUTPUT
DOMAIN_IN = toy_data_spline.DOM_INPUT
LOCS_IN = np.linspace(DOMAIN_IN[0, 0], DOMAIN_IN[0, 1], toy_data_spline.N_LOCS_INPUT)
N_EVALS_IN = 100
N_EVALS_OUT = 100
N_EVALS_FPCA = 100
PARAMS = {"regu": REGU, "kin_sigma": KIN_SIGMA, "kout_sigma": KOUT_SIGMA, "keval_sigma": KEVAL_SIGMA,
          "n_fpca": N_FPCA, "n_evals_fpca": N_EVALS_FPCA, "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
          "domain_in": DOMAIN_IN, "domain_out": DOMAIN_OUT}


# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784
SEED_INPUT = 768
SEED_OUTPUT = 456
SEED_MISSING = 560

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_INPUT)
seeds_noise_in = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_OUTPUT)
seeds_noise_out = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_MISSING)
seeds_missing = np.random.randint(100, 2000, N_AVERAGING)

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
    configs, regs = generate_expes.dti_kam(**PARAMS)
    scores_dicts = []
    Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(NSAMPLES_LIST[0], seed=784)
    Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, 586)
    Xtrain_deg = ([LOCS_IN for i in range(Xtrain_deg.shape[0])], [Xtrain_deg[i]for i in range(Xtrain_deg.shape[0])])
    Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, 765)
    Ytrain_deg = ([np.squeeze(y) for y in Ytrain_deg[0]], Ytrain_deg[1])
    Ytest = ([np.squeeze(y) for y in Ytest[0]], Ytest[1])
    Xtest = ([LOCS_IN for i in range(Xtest.shape[0])], [Xtest[i] for i in range(Xtest.shape[0])])
    best_config, best_result, score_test = parallel_tuning.parallel_tuning(
        regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
        min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((best_config, best_result, score_test), out, pickle.HIGHEST_PROTOCOL)

    # configs, regs = generate_expes.toy_spline_kpl(KER_SIGMA, REGU)
    # scores_dicts = []
    # for i in range(N_AVERAGING):
    #     scores_dicts.append({})
    #     for n_samples in NSAMPLES_LIST:
    #         Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(n_samples, seed=seeds_data[i])
    #         scores_dicts[i][n_samples] = []
    #         for deg in MISSING_LEVELS:
    #             Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
    #             Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, seeds_noise_out[i])
    #             Ytrain_deg = degradation.downsample_output(Ytrain_deg, deg, seeds_missing[i])
    #             best_config, best_result, score_test = parallel_tuning.parallel_tuning(
    #                 regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
    #                 min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
    #             scores_dicts[i][n_samples].append(score_test)
    #     with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
    #         pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)
    #
    # with open(rec_path + "/full.pkl", "wb") as out:
    #     pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)