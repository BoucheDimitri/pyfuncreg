import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 40})
plt.rcParams.update({"lines.linewidth": 4})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 2.5})
plt.rcParams.update({"xtick.major.size": 10})
plt.rcParams.update({"xtick.major.width": 2.5})
plt.rcParams.update({"ytick.major.size": 10})
plt.rcParams.update({"ytick.major.width": 2.5})

def mean_variance_result_speech(path, key):
    with open(path + "/9_" + key + ".pkl", "rb") as inp:
        _1, _2, score_test = pickle.load(inp)
    return np.mean(score_test), np.std(score_test)


path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_27-05-2020_08-55/outputs/"
KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL", "GLO", "TTCL", "TTCD")
KEYS = ("LP", "LA", "TBCL", "TBCD", "GLO", "TTCD", "TTCL", "VEL")
# KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL")

folders_speech = ["speech_kpl_rffs100_max", "speech_3be_fourier", "speech_fkrr_multi", "speech_ke_multi"]
folders_method_dict = dict()
folders_method_dict[folders_speech[0]] = "KPL"
folders_method_dict[folders_speech[1]] = "3BE"
folders_method_dict[folders_speech[2]] = "FKRR"
folders_method_dict[folders_speech[3]] = "KE"

means = {folders_method_dict[folder]: [] for folder in folders_speech}
stds = {folders_method_dict[folder]: [] for folder in folders_speech}

# Compute means and standard deviations
for key in KEYS:
    for folder in folders_speech:
        m, s = mean_variance_result_speech(path + folder, key)
        means[folders_method_dict[folder]].append(m)
        stds[folders_method_dict[folder]].append(s)

count = 0
for key in KEYS:
    best = np.min([means[folders_method_dict[folder]][count] for folder in folders_speech])
    for folder in folders_speech:
        means[folders_method_dict[folder]][count] *= 1 / best
        stds[folders_method_dict[folder]][count] *= 1 / best
    count += 1


# Plot
x = np.arange(len(KEYS))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
n_folders = len(folders_speech)
add = 0
for folder in folders_speech:
    rects = ax.bar(x - 2 * width + width/2 + add, means[folders_method_dict[folder]], width, yerr=stds[folders_method_dict[folder]],
                   capsize=5, label=folders_method_dict[folder])
    add += width

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(KEYS)
ax.legend()
ax.set_xlabel("Vocal tract")
ax.set_ylabel("MSE")