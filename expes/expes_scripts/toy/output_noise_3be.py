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
from model_eval import parallel_tuning
from expes import generate_expes

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "output_noise_3be2"
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
REGU = np.geomspace(1e-11, 1e2, 100)

DOMAIN_OUT = toy_data_spline.DOM_OUTPUT
DOMAIN_IN = toy_data_spline.DOM_INPUT
LOCS_IN = np.linspace(DOMAIN_IN[0, 0], DOMAIN_IN[0, 1], toy_data_spline.N_LOCS_INPUT)

KIN_SIGMA = [0.025]
N_FREQS_IN = [15]
N_RFFS = [200]
RFFS_SEED = [347]

NSAMPLES_LIST = [200]

NOISE_INPUT = 0.07
NOISE_OUTPUT = np.linspace(0, 1.5, 50)
N_SAMPLES = 200

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784
SEED_INPUT = 768
SEED_OUTPUT = 456

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_INPUT)
seeds_noise_in = np.random.randint(100, 2000, N_AVERAGING)
np.random.seed(SEED_OUTPUT)
seeds_noise_out = np.random.randint(100, 2000, N_AVERAGING)

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

    # ############################## Run experiment ####################################################################
    # configs, regs = generate_expes.dti_3be_fourier(KIN_SIGMA, REGU, False, N_FREQS_IN, N_FREQS_OUT,
    #                                                N_RFFS, RFFS_SEED, DOMAIN_IN, DOMAIN_OUT)
    configs, regs = generate_expes.toy_3be_fpcasplines(KIN_SIGMA, REGU, False, N_FREQS_IN, N_RFFS, RFFS_SEED)
    scores_dicts = [{} for i in range(N_AVERAGING)]
    # for i in range(N_AVERAGING):
    for i in range(N_AVERAGING):
        Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_SAMPLES, seed=seeds_data[i], squeeze_locs=True)
        Xtest = ([LOCS_IN for j in range(Xtest.shape[0])], [Xtest[j] for j in range(Xtest.shape[0])])
        # Add input noise
        Xtrain = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
        Xtrain = ([LOCS_IN for j in range(Xtrain.shape[0])], [Xtrain[j]for j in range(Xtrain.shape[0])])
        scores_dicts[i][N_SAMPLES] = []
        for noise in NOISE_OUTPUT:
            Ytrain_deg = degradation.add_noise_outputs(Ytrain, noise, seeds_noise_out[i])
            best_config, best_result, score_test = parallel_tuning.parallel_tuning(
                regs, Xtrain, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
                min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
            scores_dicts[i][N_SAMPLES].append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((NOISE_OUTPUT, scores_dicts), out, pickle.HIGHEST_PROTOCOL)

    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((NOISE_OUTPUT, scores_dicts), out, pickle.HIGHEST_PROTOCOL)