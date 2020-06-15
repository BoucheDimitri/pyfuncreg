import numpy as np
import os
import sys
import pickle
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from data import degradation
from data import toy_data_spline
from expes import generate_expes
from model_eval import parallel_tuning

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "output_missing_kam"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "discrete_general"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 2
N_PROCS = None
MIN_PROCS = 32
# N_PROCS = 7
# MIN_PROCS = None

# ############################### Experiment parameters ################################################################
# REGU = np.geomspace(1e-8, 1, 50)
# REGU = np.geomspace(1e-11, 1e2, 500)
REGU = [1e-8, 1e-6]
KIN_SIGMA = [0.25]
KOUT_SIGMA = [0.1]
KEVAL_SIGMA = [2.5]
N_FPCA = [20]
NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02
MISSING_LEVELS = np.arange(0, 1, 0.05)
# MISSING_LEVELS = [0.95]
# MISSING_LEVELS = [0.9]
N_SAMPLES = 200
DOMAIN_OUT = toy_data_spline.DOM_OUTPUT
DOMAIN_IN = toy_data_spline.DOM_INPUT
LOCS_IN = np.linspace(DOMAIN_IN[0, 0], DOMAIN_IN[0, 1], toy_data_spline.N_LOCS_INPUT)
N_EVALS_IN = 210
N_EVALS_OUT = 210
N_EVALS_FPCA = 210
PARAMS = {"regu": REGU, "kin_sigma": KIN_SIGMA, "kout_sigma": KOUT_SIGMA, "keval_sigma": KEVAL_SIGMA,
          "n_fpca": N_FPCA, "n_evals_fpca": N_EVALS_FPCA, "n_evals_in": N_EVALS_IN, "n_evals_out": N_EVALS_OUT,
          "domain_in": DOMAIN_IN, "domain_out": DOMAIN_OUT}

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 1
# N_AVERAGING = 10
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
#
#
# Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_SAMPLES, seed=784)
# Xtest = ([LOCS_IN for j in range(Xtest.shape[0])], [Xtest[j] for j in range(Xtest.shape[0])])
# Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, 675)
# Xtrain_deg = ([LOCS_IN for j in range(Xtrain_deg.shape[0])], [Xtrain_deg[j]for j in range(Xtrain_deg.shape[0])])
# Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, 743)
# Ytrain_deg = degradation.downsample_output(Ytrain_deg, MISSING_LEVELS[0], 342)
#
# configs, regs = generate_expes.dti_kam(**PARAMS)
#
# reg = regs[0]
#
# reg.fit(Xtrain_deg, Ytrain_deg)
#
# #
# # preds = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
# #
# # Klocs_out = reg.kernel_out(np.expand_dims(Ytest[0][0], axis=1), np.expand_dims(reg.space_out, axis=1))
#
# # Ycentered_func, Ymean = reg.fit(Xtrain_deg, Ytrain_deg)
# #
# # reg.fpca.fit(Ycentered_func)
# # Yfpca = reg.fpca.get_regressors(reg.n_fpca)
# #
# # Yfpca_evals = np.array([f(reg.space_out) for f in Yfpca])
# #
# # reg.fpca.fit(Ycentered_func)
# #
# # Yevals = [y(reg.space_out) for y in Ycentered_func]
# # # Ymean_eval = Ymean(reg.space_out)
#
# # Ycentered, Ymean = reg.fit(Xtrain_deg, Ytrain_deg)
# #
# # Yevals = [y(reg.space_out) for y in Ycentered]
# # Ymean_eval = Ymean(reg.space_out)
# # Yfpca, kernel_out, space_out, domain_out = reg.fit(Xtrain_deg, Ytrain_deg)
# #
# # Klocs_out = kernel_out(np.expand_dims(space_out, axis=1), np.expand_dims(space_out, axis=1))
# # n_fpca = len(Yfpca)
# # Yfpca_evals = np.array([f(space_out) for f in Yfpca])
# # test = Yfpca[0](space_out)
#


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
    for i in range(N_AVERAGING):
        scores_dicts.append({})
        Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_SAMPLES, seed=seeds_data[i])
        scores_dicts[i][N_SAMPLES] = []
        Xtest = ([LOCS_IN for j in range(Xtest.shape[0])], [Xtest[j] for j in range(Xtest.shape[0])])
        for deg in MISSING_LEVELS:
            Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
            Xtrain_deg = ([LOCS_IN for j in range(Xtrain_deg.shape[0])], [Xtrain_deg[j]for j in range(Xtrain_deg.shape[0])])
            Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, seeds_noise_out[i])
            Ytrain_deg = degradation.downsample_output(Ytrain_deg, deg, seeds_missing[i])
            best_config, best_result, score_test = parallel_tuning.parallel_tuning(
                regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
                min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
            scores_dicts[i][N_SAMPLES].append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)

    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)