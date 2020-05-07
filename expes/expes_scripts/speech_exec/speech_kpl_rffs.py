import numpy as np
import os
import sys
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)
# path = os.getcwd()

# Local importss
from expes import generate_expes, run_expes

# ############################### Execution config #####################################################################
# Path to the data
DATA_PATH = path + "/data/dataspeech/raw/"
# Record config
OUTPUT_FOLDER = "/speech_kpl_rffs2"

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
REGU_GRID = list(np.geomspace(1e-11, 1e-4, 50))
# REGU_GRID = [1e-10, 1e-7]
# N_FREQS = [5]
# Standard deviation parameter for the input kernel
KER_SIGMA = 1
# Decrease base
# DECREASE_BASE = np.arange(1, 1.6, 0.1)
DECREASE_BASE = 1
# Number of evaluations for FPCA
N_RFFS = [100]
CENTER_OUTPUT = [True, False]
# RFFS_SIGMA
RFFS_SIGMA = [10, 25, 50, 75, 100]
# RFFS_SIGMA = [0.025, 0.05]

# Seeds for averaging of expes (must all be of the same size)
N_AVERAGING = 10
# N_AVERAGING = 2
SEED_DATA = 784
SEED_RFF = 567

# Generate seeds
np.random.seed(SEED_DATA)
seeds_data = np.random.randint(100, 2000, N_AVERAGING)

if __name__ == '__main__':

    # Create folder for saving results
    rec_path = run_expes.create_output_folder(path, OUTPUT_FOLDER)
    # Generate configurations and corresponding regressors
    configs, regs = generate_expes.speech_rffs_kpl(KER_SIGMA, REGU_GRID, N_RFFS, RFFS_SIGMA,
                                                   SEED_RFF, CENTER_OUTPUT, DOMAIN)
    # Run expes
    best_configs, best_results, scores_test = run_expes.run_expe_speech(
        configs, regs, seeds=seeds_data, data_path=DATA_PATH, rec_path=rec_path,
        input_indexing=INPUT_INDEXING, output_indexing=OUTPUT_INDEXING, n_folds=N_FOLDS,
        n_procs=N_FOLDS, min_nprocs=MIN_PROCS)