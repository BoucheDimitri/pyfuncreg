import pickle
import matplotlib.pyplot as plt
import numpy as np

from data import toy_data_spline

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_16-04-2020_11-33/outputs/output_noise_multi/"
all_results = []
for i in range(10):
    with open(path + str(i) + ".pkl", "rb") as inp:
        noise_levels, scores = pickle.load(inp)
        all_results.append(scores[-1])

with open(path + "9.pkl", "rb") as inp:
    noise_levels, scores = pickle.load(inp)
scores = scores[9]

# Estimate mean absolute signal
Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(5000)
mean_abs_signal = np.mean(np.abs(np.array(Ytrain[1])))

if noise_levels[0] == 0:
    beg_ind = 1
else:
    beg_ind = 0
snr_grid = np.flip((mean_abs_signal / noise_levels[beg_ind:]))

plt.figure()
sub_n_samples = [10, 50, 100, 500]
for key in sub_n_samples:
    concat_results = []
    mean_results = []
    std_results = []
    for i in range(10):
        print(all_results[i][key])
        results_test = np.flip(np.array(all_results[i][key][beg_ind:]))
        concat_results.append(results_test)

sub_n_samples = [10, 50, 100, 500]
for key in sub_n_samples:
    results_test = np.flip(np.array(scores[key][beg_ind:]))
    plt.semilogy(snr_grid, results_test, label="N=" + str(key), marker="o")
plt.legend()