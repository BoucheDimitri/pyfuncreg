import os
import pickle
import numpy as np

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-06-2020_09-47/outputs/"

def consolidate_dicts(folder, path=path):
    results = []
    for i in range(10):
        with open(path + folder + str(i) + ".pkl", "rb") as inp:
            missing_levels, result_dict = pickle.load(inp)
        results.append(result_dict[i][200])
    return np.array(results)


folders = ["output_missing_kpl/", "output_missing_fkrr/", "output_missing_3be/"]
corresp = ["KPL", "FKRR", "3BE"]

all_results = {corresp[i]: np.array(consolidate_dicts(folders[i])) for i in range(len(folders))}
means = {key: np.mean(all_results[key], axis=0) for key in all_results.keys()}



def get_mean_std(folder, n_samp)


with open(path + "full.pkl", "rb") as inp:
    missing_levels, big_list = pickle.load(inp)

n_samples = list(big_list[0].keys())

array_dict = {n: [] for n in n_samples}
means_dict = {}
stds_dict = {}

for n in n_samples:
    for i in range(len(big_list)):
        array_dict[n].append(big_list[i][n])

for n in n_samples:
    array_dict[n] = np.array(array_dict[n])

for n in n_samples:
    stds_dict[n] = array_dict[n].std(axis=0)
    means_dict[n] = array_dict[n].mean(axis=0)

sub_n_samples = [10, 50, 100, 500]
markers = ["o", "v", "s", "D"]
count = 0
for n in sub_n_samples:
    # plt.plot(missing_levels, means_dict[n], label="N=" + str(n))
    plt.errorbar(100 * np.array(missing_levels), means_dict[n],
                 yerr=stds_dict[n], marker=markers[count], capsize=5, label="N=" + str(n), lolims=True)
    count += 1

plt.legend()
plt.ylabel("MSE")
plt.xlabel("Percentage missing")

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-06-2020_09-47/outputs/"

