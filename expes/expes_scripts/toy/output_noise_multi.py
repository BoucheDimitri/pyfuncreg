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
OUTPUT_FOLDER = "output_noise_multi"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "output_noise"

# ############################### Experiment parameters ################################################################
KER_SIGMA = 20
REGU = np.geomspace(1e-11, 1e2, 500)
# REGU = np.geomspace(1e-9, 1, 10)
NOISE_INPUT = 0.07
NOISE_OUTPUT = np.linspace(0, 1.5, 50)
# NOISE_OUTPUT = np.linspace(0, 1.5, 10)
NSAMPLES_LIST = [10, 25, 50, 100, 250, 500, 1000]
# NSAMPLES_LIST = [10, 25, 50]
# NOISE_OUTPUT = np.linspace(0, 1.5, 3)
INPUT_INDEXING = "array"
OUTPUT_INDEXING = "discrete_general"
N_FOLDS = 5
N_PROCS = None
MIN_PROCS = 32
# N_PROCS = 7
# MIN_PROCS = None

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
    configs, regs = generate_expes.toy_spline_kpl(KER_SIGMA, REGU)
    scores_dicts = []
    # for i in range(N_AVERAGING):
    for i in range(9, 10):
        scores_dicts.append({})
        for n_samples in NSAMPLES_LIST:
            Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(n_samples, seed=seeds_data[i])
            # Add input noise
            Xtrain = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, seeds_noise_in[i])
            scores_dicts[i][n_samples] = []
            for noise in NOISE_OUTPUT:
                Ytrain_deg = degradation.add_noise_outputs(Ytrain, noise, seeds_noise_out[i])
                best_config, best_result, score_test = parallel_tuning.parallel_tuning(
                    regs, Xtrain, Ytrain_deg, Xtest, Ytest, configs=configs, n_folds=N_FOLDS, n_procs=N_PROCS,
                    min_nprocs=MIN_PROCS, input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING)
                scores_dicts[i][n_samples].append(score_test)
        with open(rec_path + "/" + str(i) + ".pkl", "wb") as out:
            pickle.dump((NOISE_OUTPUT, scores_dicts), out, pickle.HIGHEST_PROTOCOL)

    with open(rec_path + "/full.pkl", "wb") as out:
        pickle.dump((NOISE_OUTPUT, scores_dicts), out, pickle.HIGHEST_PROTOCOL)