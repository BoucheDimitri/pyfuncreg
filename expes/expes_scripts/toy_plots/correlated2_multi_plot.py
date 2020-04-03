import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

fontweight = 540
plt.rcParams.update({'ps.useafm': True})
plt.rcParams.update({'pdf.use14corefonts': True})
plt.rcParams.update({'text.usetex':True})
plt.rcParams.update({"font.size": 32})
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.weight": fontweight})
plt.rcParams.update({"lines.linewidth": 7})
plt.rcParams.update({"lines.markersize": 15})
plt.rcParams.update({"axes.linewidth": 2.5})
plt.rcParams.update({"xtick.major.size": 10})
plt.rcParams.update({"xtick.major.width": 2.5})
plt.rcParams.update({"ytick.major.size": 10})
plt.rcParams.update({"ytick.major.width": 2.5})

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_01-04-2020_09-02/outputs/toy_correlated_multi/"
# path = "/home/dimitri/Desktop/Telecom/nonlinear_functional_regressions/outputs/toy_correlated/"

with open(path + "3.pkl", "rb") as inp:
    scores_dicts, scores_dicts_corr = pickle.load(inp)

scores, scores_corr = [], []

for i in range(len(scores_dicts)):
    scores.append([scores_dicts[i][key][0] for key in scores_dicts[i].keys()])
    scores_corr.append([scores_dicts_corr[i][key][0] for key in scores_dicts[i].keys()])

n_samples = list(scores_dicts[0].keys())

scores = np.array(scores)
scores_corr = np.array(scores_corr)

scores_mean = np.mean(scores, axis=0)
scores_corr_mean = np.mean(scores_corr, axis=0)
scores_std = np.std(scores, axis=0)

plt.errorbar(n_samples, scores_corr_mean, yerr=scores_std, marker="o", capsize=3)