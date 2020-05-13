import numpy as np
import os
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local imports
from expes import generate_expes, run_expes

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/raw/"
# Record config
OUTPUT_FOLDER = "/speech_fkrr_gauss"
# Indexing
INPUT_INDEXING = "list"
OUTPUT_INDEXING = "discrete_general"
# Number of folds
N_FOLDS = 5

# Exec config
# N_PROCS = 7
# MIN_PROCS = None
N_PROCS = None
MIN_PROCS = 32

# ############################### Regressor config #####################################################################
# Output domain
DOMAIN = np.array([[0, 1]])
# Regularization parameters grid
REGU_GRID = list(np.geomspace(1e-10, 1e-3, 50))
# REGU_GRID = [1e-10, 1e-7]
# Standard deviation parameter for the input kernel
KIN_SIGMA = 1
# KOUT_SIGMA = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15]
KOUT_SIGMA = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
# Approximation locations
APPROX_LOCS = np.linspace(0, 1, 300)
CENTER_OUTPUT = [False, True]

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
SEED_DATA = 784

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)

if __name__ == '__main__':

    # Create folder for saving results
    rec_path = run_expes.create_output_folder(path, OUTPUT_FOLDER)
    # Generate configurations and corresponding regressors
    configs, regs = generate_expes.speech_fkrr_gauss(KIN_SIGMA, KOUT_SIGMA, REGU_GRID, APPROX_LOCS, CENTER_OUTPUT)
    # Run expes
    best_configs, best_results, scores_test = run_expes.run_expe_speech(
        configs, regs, seeds=seeds_data, data_path=DATA_PATH, rec_path=rec_path,
        input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING, n_folds=N_FOLDS,
        n_procs=N_FOLDS, min_nprocs=MIN_PROCS)