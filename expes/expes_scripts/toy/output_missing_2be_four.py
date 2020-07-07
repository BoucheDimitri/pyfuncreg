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
OUTPUT_FOLDER = "output_missing_2be_four"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
# Shuffle seed
SHUFFLE_SEED = 784
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
N_PROCS = None
MIN_PROCS = 32
# N_PROCS = 7
# MIN_PROCS = None

# ############################### Experiment parameters ################################################################
REGU = np.geomspace(1e-11, 1e2, 100)
# REGU = np.geomspace(1e-9, 1, 10)
NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02
NSAMPLES_LIST = [200]
MISSING_LEVELS = np.arange(0, 1, 0.05)
# NSAMPLES_LIST = [20, 50]
# MISSING_LEVELS = [0, 0.1]
DOMAIN_OUT = toy_data_spline.DOM_OUTPUT
DOMAIN_IN = toy_data_spline.DOM_INPUT
LOCS_IN = np.linspace(DOMAIN_IN[0, 0], DOMAIN_IN[0, 1], toy_data_spline.N_LOCS_INPUT)

KER_SIGMA = 20
FREQS_OUT = [20]

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 2
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
    # configs, regs = generate_expes.toy_2be(KER_SIGMA, REGU, DOMAIN_OUT)
    configs, regs = generate_expes.toy_2be_four(KER_SIGMA, REGU, FREQS_OUT, DOMAIN_OUT)
    # configs, regs = generate_expes.toy_3be_fpcasplines(KIN_SIGMA, REGU, False, N_FREQS_IN, N_RFFS, RFFS_SEED)
    scores_dicts = [{} for i in range(N_AVERAGING)]
    for i in range(N_AVERAGING):
        scores_dicts.append({})
        for n_samples in NSAMPLES_LIST:
            Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(n_samples, seed=seeds_data[i],
                                                                        squeeze_locs=True)
            scores_dicts[i][n_samples] = []
            for deg in MISSING_LEVELS:
                Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
                Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, seeds_noise_out[i])
                Ytrain_deg = degradation.downsample_output(Ytrain_deg, deg, seeds_missing[i])
                best_config, best_result, score_test = parallel_tuning.parallel_tuning(
                    regs, Xtrain_deg, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
                    min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
                scores_dicts[i][n_samples].append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)

    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((MISSING_LEVELS, scores_dicts), out, pickle.HIGHEST_PROTOCOL)