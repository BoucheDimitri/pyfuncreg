import numpy as np
import os
import sys
import pickle
import pathlib
import matplotlib.pyplot as plt

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)

# Local imports
from data import toy_data_spline

# Path to pickle file
PICKLE_FILE = path + "/outputs/output_noise/output_noise.pkl"

# Config
N_TRAIN = 500

if __name__ == '__main__':
    with open(PICKLE_FILE, "rb") as inp:
        noise_levels, scores_test = pickle.load(inp)

    # Load dataset to compute mean abs signal (the seed set in the file toy_data_spline.py ensures
    # that this is the same dataset as the one used to generate the pickle file
    Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_TRAIN)
    mean_abs_signal = np.mean(np.abs(np.array(Ytrain[1])))
    # Compute SNR grids
    if noise_levels[0] == 0:
        beg_ind = 1
    else:
        beg_ind = 0
    snr_grid = np.flip((mean_abs_signal / noise_levels[beg_ind:]))
    results_test = np.flip(np.array(scores_test[beg_ind:]))
    # Generate plot
    plt.figure()
    plt.plot(snr_grid, results_test, marker="o")
    plt.xlabel("Signal to noise ratio")
    plt.ylabel("MSE score on test set")
    plt.show()

