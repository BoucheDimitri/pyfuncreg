import pickle
import matplotlib.pyplot as plt
import numpy as np

from data import toy_data_spline

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/output_noise/"
with open(path + "full.pkl", "rb") as inp:
    noise_levels, scores = pickle.load(inp)

# Estimate mean absolute signal
Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(5000)
mean_abs_signal = np.mean(np.abs(np.array(Ytrain[1])))

if noise_levels[0] == 0:
    beg_ind = 1
else:
    beg_ind = 0
snr_grid = np.flip((mean_abs_signal / noise_levels[beg_ind:]))

plt.figure()
for key in scores.keys():
    results_test = np.flip(np.array(scores[key][beg_ind:]))
    plt.semilogy(snr_grid, results_test, label="N=" + str(key), marker="o")
plt.legend()