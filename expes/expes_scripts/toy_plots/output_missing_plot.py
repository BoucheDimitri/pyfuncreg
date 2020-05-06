import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# plt.style.use('seaborn')
# rc('text.usetext')


fontweight = 500
# plt.rcParams.update({'ps.useafm': True})
# plt.rcParams.update({'pdf.use14corefonts': True})
# plt.rcParams.update({'text.usetex': True})
# plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 32})
# plt.rcParams.update({"mathtext.bf":"sans:bold"})
# plt.rcParams.update({"mathtext.fontset": "cm"})
# plt.rcParams.update({"pdf.fonttype": 1})
# plt.rcParams.update({'pdf.use14corefonts': True})
plt.rcParams.update({"font.weight": fontweight})
# plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({"lines.linewidth": 7})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 2.5})
plt.rcParams.update({"xtick.major.size": 10})
plt.rcParams.update({"xtick.major.width": 2.5})
plt.rcParams.update({"ytick.major.size": 10})
plt.rcParams.update({"ytick.major.width": 2.5})


# path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/output_missing2/"
# # path = "/home/dimitri/Desktop/Telecom/nonlinear_functional_regressions/outputs/toy_correlated/"
# with open(path + "full.pkl", "rb") as inp:
#     missing_levels, scores = pickle.load(inp)
#
# plt.figure()
# for key in scores.keys():
#     plt.plot(missing_levels, scores[key], label="N=" + str(key))
# plt.legend()

# MULTI
path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_06-04-2020_09-42/outputs/output_missing_multi/"

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
for n in sub_n_samples:
    # plt.plot(missing_levels, means_dict[n], label="N=" + str(n))
    plt.errorbar(100 * np.array(missing_levels), means_dict[n],
                 yerr=stds_dict[n], marker="o", capsize=3, label="N=" + str(n), lolims=True)

plt.legend()
plt.ylabel("MSE")
plt.xlabel("Output degradation (% of evaluations missing)")

