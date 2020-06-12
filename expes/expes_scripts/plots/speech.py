import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 40})
plt.rcParams.update({"lines.linewidth": 5})
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


path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_10-06-2020_09-15/outputs/"
KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL", "GLO", "TTCL", "TTCD")
# KEYS = ("LP", "LA", "TBCL", "TBCD", "GLO", "VEL")
# KEYS = ("LP", "LA", "TBCL", "TBCD", "GLO", "TTCD", "VEL")
# KEYS = ("LA", "TBCD", "TTCD", "VEL")

# folders_speech = ["speech_kpl_rffs75_max", "speech_3be_fourier_morefreqs", "speech_fkrr_multi" , "speech_ke_multi"]
# folders_speech = ["speech_kpl_rffs100_max", "speech_3be_fourier_morefreqs", "speech_fkrr_multi"]#, "speech_ke_multi"]
# folders_speech = ["speech_kpl_rffs75_missing", "speech_3be_fourier_missing", "speech_fkrr_missing", "speech_ke_multi"]
# folders_speech = ["speech_kpl_rffs75_missing", "speech_3be_fourier_missing", "speech_fkrr_missing"]
folders_speech = ["speech_fkrr_multi", "speech_fkrr_eigsolve"]
folders_method_dict = dict()
# folders_method_dict[folders_speech[0]] = "KPL"
# folders_method_dict[folders_speech[1]] = "3BE"
# folders_method_dict[folders_speech[2]] = "FKRR"
folders_method_dict[folders_speech[0]] = "FKRR Syl"
folders_method_dict[folders_speech[1]] = "FKRR Eigapprox"
# folders_method_dict[folders_speech[3]] = "KE"

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


# Plot vert
x = np.arange(len(KEYS))  # the label locations
width = 0.2 # the width of the bars
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# cycle_modif = cycle[0:3] + [cycle[4]]
cycle_modif = [cycle[2], cycle[6]]
# cycle_modif = cycle[0:4]

# fig, ax = plt.subplots()
ax = plt.subplot(gs[0])
n_folders = len(folders_speech)
add = 0
error_kw = dict(lw=6, capsize=6, capthick=4)
count = 0
for folder in folders_speech:
    # rects = ax.bar(x - 2 * width + width/2 + add, means[folders_method_dict[folder]], width, yerr=stds[folders_method_dict[folder]],
    #                error_kw=error_kw, label=folders_method_dict[folder], color=cycle_modif[count])
    rects = ax.bar(x - 1 * width/2 + add, means[folders_method_dict[folder]], width, yerr=stds[folders_method_dict[folder]],
                   error_kw=error_kw, label=folders_method_dict[folder],  color=cycle_modif[count])
    add += width
    count += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(KEYS)
ax.legend()
ax.set_xlabel("Vocal tract")
ax.set_ylabel("normalized MSE")
ax.set_ylim(0.827, 1.3)
# ax.set_yscale("log")


# Plot horizontal
y = np.arange(len(KEYS))  # the label locations
width = 0.225  # the width of the bars

fig, ax = plt.subplots()
n_folders = len(folders_speech)
add = 0
error_kw = dict(lw=5, capsize=5, capthick=3)
for folder in folders_speech:
    rects = ax.bar(x - 2 * width + width/2 + add, means[folders_method_dict[folder]], width, yerr=stds[folders_method_dict[folder]],
                   error_kw=error_kw, label=folders_method_dict[folder])
    # rects = ax.barh(y - 1 * width + add, means[folders_method_dict[folder]], width, xerr=stds[folders_method_dict[folder]],
    #                error_kw=error_kw, label=folders_method_dict[folder])
    add += width

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_yticks(y)
ax.set_yticklabels(KEYS)
ax.legend()
ax.set_ylabel("Vocal tract")
ax.set_xlabel("normalized MSE")
ax.set_xlim(0.827)
# ax.set_yscale("log")