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

def mean_variance_time_speech(path, keys):
    timers_total = []
    for key in keys:
        with open(path + "/9_" + key + ".pkl", "rb") as inp:
            timers_total += pickle.load(inp)
    return np.mean(timers_total), np.std(timers_total)

path = os.getcwd() + "/outputs/"
KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL", "GLO", "TTCL", "TTCD")
# KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL")

folders_speech = ["speech_kpl_rffs75_timer", "speech_3be_fourier_timer", "speech_fkrr_timer"]
folders_method_dict = dict()
folders_method_dict[folders_speech[0]] = "KPL"
folders_method_dict[folders_speech[1]] = "3BE"
folders_method_dict[folders_speech[2]] = "FKRR"
# folders_method_dict[folders_speech[3]] = "KE"

means = {folders_method_dict[folder]: [] for folder in folders_speech}
stds = {folders_method_dict[folder]: [] for folder in folders_speech}

# Compute means and standard deviations
for folder in folders_speech:
    m, s = mean_variance_time_speech(path + folder, KEYS)
    means[folders_method_dict[folder]].append(m)
    stds[folders_method_dict[folder]].append(s)

# count = 0
# for key in KEYS:
#     best = np.min([means[folders_method_dict[folder]][count] for folder in folders_speech])
#     for folder in folders_speech:
#         means[folders_method_dict[folder]][count] *= 1 / best
#         stds[folders_method_dict[folder]][count] *= 1 / best
#     count += 1

methods = ["KPL", "3BE", "FKRR"]
y_pos = np.arange(len(methods))
perfs = [means[method][0] for method in methods]
errs = [stds[method][0] for method in methods]

# Plot
width = 0.1  # the width of the bars

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()
error_kw = dict(lw=5, capsize=5, capthick=3)
ax.barh(y_pos, perfs, xerr=errs, align='center', color=cycle[0:3], error_kw=error_kw)
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('CPU time', labelpad=2)

ax.legend()
ax.set_ylabel("CPU time")