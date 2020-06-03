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


def mean_variance_time_dti(path):
    with open(path + "/9.pkl", "rb") as inp:
        timers = pickle.load(inp)
    return np.mean(timers), np.std(timers)


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
folders_method_speech = dict()
folders_method_speech[folders_speech[0]] = "KPL"
folders_method_speech[folders_speech[1]] = "3BE"
folders_method_speech[folders_speech[2]] = "FKRR"
# folders_method_dict[folders_speech[3]] = "KE"

folders_dti = ["dti_kpl_timer", "dti_3be_timer", "dti_fkrr_timer", "dti_kam_timer"]
folders_method_dti = dict()
folders_method_dti[folders_dti[0]] = "KPL"
folders_method_dti[folders_dti[1]] = "3BE"
folders_method_dti[folders_dti[2]] = "FKRR"
folders_method_dti[folders_dti[3]] = "KAM"

means_speech = {folders_method_speech[folder]: [] for folder in folders_speech}
stds_speech = {folders_method_speech[folder]: [] for folder in folders_speech}

means_dti = {folders_method_dti[folder]: [] for folder in folders_dti}
stds_dti = {folders_method_dti[folder]: [] for folder in folders_dti}

# Compute means and standard deviations
for folder in folders_speech:
    m, s = mean_variance_time_speech(path + folder, KEYS)
    means_speech[folders_method_speech[folder]].append(m)
    stds_speech[folders_method_speech[folder]].append(s)

# Compute means and standard deviations
for folder in folders_dti:
    m, s = mean_variance_time_dti(path + folder)
    means_dti[folders_method_dti[folder]].append(m)
    stds_dti[folders_method_dti[folder]].append(s)

error_kw = dict(lw=6, capsize=6, capthick=4)
width = 0.2  # the width of the bars
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# cycle_modif = cycle[0:3] + [cycle[4]]
cycle_modif = cycle[0:4]


# Horizontal bars
fig, ax = plt.subplots()

yticks = np.arange(0, 2)
yticks_text = ["DTI", "Speech"]

add = 0
count = 0
for folder in folders_dti:
    rects = ax.barh(np.array([0]) - 2 * width + width/2 + add,
                      means_dti[folders_method_dti[folder]], width,
                      xerr=stds_dti[folders_method_dti[folder]],
                      error_kw=error_kw, label=folders_method_dti[folder], color=cycle_modif[count])
    add += width
    count += 1

add = 0
count = 0
for folder in folders_speech:
    rects = ax.barh(np.array([1]) - 1 * width + add,
                   means_speech[folders_method_speech[folder]], width,
                   xerr=stds_speech[folders_method_speech[folder]],
                   error_kw=error_kw, color=cycle[count])
    add += width
    count += 1

ax.legend()
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_text)
ax.set_xscale('log')
ax.set_xlabel("CPU time (log scale)")




# Vertical bars
fig, ax = plt.subplots()

xticks = np.arange(0, 2)
xticks_text = ["DTI", "Speech"]

add = 0
count = 0
for folder in folders_dti:
    rects = ax.bar(np.array([0]) - 2 * width + width/2 + add,
                      means_dti[folders_method_dti[folder]], width,
                      yerr=stds_dti[folders_method_dti[folder]],
                      error_kw=error_kw, label=folders_method_dti[folder], color=cycle_modif[count])
    add += width
    count += 1

add = 0
count = 0
for folder in folders_speech:
    rects = ax.bar(np.array([1]) - 1 * width + add,
                   means_speech[folders_method_speech[folder]], width,
                   yerr=stds_speech[folders_method_speech[folder]],
                   error_kw=error_kw, color=cycle[count])
    add += width
    count += 1

ax.legend()
ax.set_xticks(xticks)
ax.set_xticklabels(xticks_text)
ax.set_yscale('log')
ax.set_ylabel("CPU time (log scale)")



#
#
# fig, axes = plt.subplots(nrows=2)
# y = np.array([0])  # the label locations
#
# # Plot DTI
# y_dti = ["DTI"]
# add = 0
# for folder in folders_dti:
#     rects = axes[0].barh(y - 2 * width + width/2 + add,
#                          means_dti[folders_method_dti[folder]], width,
#                          xerr=stds_dti[folders_method_dti[folder]],
#                          error_kw=error_kw, label=folders_method_dti[folder])
#     add += width
#
# axes[0].set_yticks(y)
# axes[0].set_yticklabels(y_dti)
#
# # Plot speech
# y_speech = ["Speech"]
# add = 0
# for folder in folders_speech:
#     rects = axes[1].barh(y - 1 * width + add,
#                          means_speech[folders_method_speech[folder]], width,
#                          xerr=stds_speech[folders_method_speech[folder]],
#                          error_kw=error_kw)
#     add += width
#
# axes[1].set_yticks(y)
# axes[1].set_yticklabels(y_speech)
# axes[1].set_ylabel("CPU time")