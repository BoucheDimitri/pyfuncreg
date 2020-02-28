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
import expes.DEPRECATED.expes_scripts.toy.output_missing as output_missing

# Inherits config from output_missing.py which generated the pickle file
N_SAMPLES_LIST = output_missing.NSAMPLES_LIST
MISSING_LEVELS = output_missing.MISSING_LEVELS

# Path to pickle file
PICKLE_FILE = path + "/outputs/output_missing/output_missing.pkl"

if __name__ == '__main__':
    results_dict = {miss: [] for miss in MISSING_LEVELS}
    results_dict_nsamples = {nsamp: [] for nsamp in N_SAMPLES_LIST}

    with open(PICKLE_FILE, "rb") as inp:
        input_config, scores_test = pickle.load(inp)

    count = 0
    for deg in MISSING_LEVELS:
        for n_samples in N_SAMPLES_LIST:
            for config in input_config:
                if config[1] == deg and config[0] == n_samples:
                    results_dict_nsamples[n_samples].append(scores_test[count])
            count += 1

    plt.figure()
    for nsamp in N_SAMPLES_LIST:
        plt.plot(100 * np.array(MISSING_LEVELS), results_dict_nsamples[nsamp], label="N=" + str(nsamp), marker="o")
    plt.legend()
    plt.ylabel("MSE score on test set")
    plt.xlabel("Output degradation (\% of evaluations missing)")
    plt.show()


