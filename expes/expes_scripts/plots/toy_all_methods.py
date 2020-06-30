import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from data import toy_data_spline

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-06-2020_09-47/outputs/"

def consolidate_dicts(folder, path=path):
    results = []
    for i in range(9):
        with open(path + folder + str(i) + ".pkl", "rb") as inp:
            missing_levels, result_dict = pickle.load(inp)
        results.append(result_dict[i][200])
    return np.array(results)


# ########################### OUTPUT MISSING ###########################################################################
# folders = ["output_missing_kpl/", "output_missing_fkrr/", "output_missing_3be/"]
# corresp = ["KPL", "FKRR", "3BE", "KAM"]
folders = ["output_missing_kpl/", "output_missing_fkrr/", "output_missing_kam/"]
corresp = ["KPL", "FKRR", "KAM"]

all_results = {corresp[i]: np.array(consolidate_dicts(folders[i])) for i in range(len(folders))}
means = {key: np.mean(all_results[key], axis=0) for key in all_results.keys()}
stds = {key: np.std(all_results[key], axis=0) for key in all_results.keys()}
missing_levels = np.arange(0, 1, 0.05)

markers = ["o", "v", "s", "D"]
count = 0
for key in corresp:
    plt.errorbar(100 * np.array(missing_levels), means[key],
                 yerr=stds[key], marker=markers[count], capsize=5, label=key, lolims=True)
    count += 1


# ########################### OUTPUT NOISE #############################################################################
folders = ["output_noise_kpl/", "output_noise_fkrr/"]
corresp = ["KPL", "FKRR"]

all_results = {corresp[i]: np.array(consolidate_dicts(folders[i])) for i in range(len(folders))}
means = {key: np.mean(all_results[key], axis=0) for key in all_results.keys()}
stds = {key: np.std(all_results[key], axis=0) for key in all_results.keys()}
noise_levels = np.linspace(0, 1.5, 50)

# Estimate mean absolute signal
Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(5000)
mean_abs_signal = np.mean(np.abs(np.array(Ytrain[1])))

if noise_levels[0] == 0:
    beg_ind = 1
else:
    beg_ind = 0
snr_grid = np.flip((mean_abs_signal / noise_levels[beg_ind:]))

markers = ["o", "v", "s", "D"]
count = 0
for key in corresp:
    plt.plot(snr_grid, np.flip(means[key][beg_ind:]), label=key, marker=markers[count])
    count += 1

plt.yscale("log")