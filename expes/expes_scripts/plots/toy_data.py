from data import toy_data_spline
from data import degradation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"font.size": 32})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})
plt.rcParams.update({"axes.linewidth": 2.5})
plt.rcParams.update({"xtick.major.size": 10})
plt.rcParams.update({"xtick.major.width": 2.5})
plt.rcParams.update({"ytick.major.size": 10})
plt.rcParams.update({"ytick.major.width": 2.5})

n_samples = 100
noise_input = 0.07
noise_output = 0.01
Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(n_samples, seed=784)

Xtrain = degradation.add_noise_inputs(Xtrain, noise_input, seed=678)
Ytrain = degradation.add_noise_outputs(Ytrain, noise_output, seed=678)
Ytrain = degradation.downsample_output(Ytrain, 0.5, 678)

dom_in = np.linspace(toy_data_spline.DOM_INPUT[0, 0], toy_data_spline.DOM_INPUT[0, 1], Xtrain.shape[1])

fig, axes = plt.subplots(2, 4, sharey="row")
for i in range(4):
    axes[0, i].plot(dom_in, Xtrain[i])
    axes[0, i].set_xlabel("$\gamma$")
    axes[1, i].scatter(Ytrain[0][i], Ytrain[1][i])
    axes[1, i].set_xlabel("$\\theta$")

axes[0, 0].set_ylabel("$x(\gamma)$")
axes[1, 0].set_ylabel("$y(\\theta)$")

t = np.array([0, 0.5, 1, 1.5, 2])
for i in range(0, 10):
    bs = BSpline.basis_element(t + i, extrapolate=False)
# sp = np.linspace(0, 4, 200)
    sp = np.linspace(0, 11, 500)
    plt.plot(sp, bs(sp))
    plt.xlabel("$\\theta$")
    plt.ylabel("$\phi(\\theta)$")