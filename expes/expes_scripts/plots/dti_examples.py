from data import loading
from data import processing
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 32})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 2.5})
plt.rcParams.update({"xtick.major.size": 10})
plt.rcParams.update({"xtick.major.width": 2.5})
plt.rcParams.update({"ytick.major.size": 10})
plt.rcParams.update({"ytick.major.width": 2.5})

path = os.getcwd()
cca, rcst = loading.load_dti(path + "/data/dataDTI/", shuffle_seed=None)
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti(cca, rcst)

fig, axes = plt.subplots(2, 4, sharey="row", sharex="col")
indlist = np.array([8, 1, 16, 0])
for i in range(4):
    axes[0, i].plot(Xtrain[0][indlist[i]], Xtrain[1][indlist[i]])
    axes[0, i].set_xlabel("$\gamma$")
    axes[1, i].plot(Ytrain[0][indlist[i]], Ytrain[1][indlist[i]])
    axes[1, i].set_xlabel("$\\theta$")

axes[0, 0].set_ylabel("$x(\gamma)$")
axes[1, 0].set_ylabel("$y(\\theta)$")

