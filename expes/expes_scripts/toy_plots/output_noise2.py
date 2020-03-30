import pickle
import matplotlib.pyplot as plt
import numpy as np

from data import toy_data_spline

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/output_noise/"
with open(path + "full.pkl", "rb") as inp:
    noise_levels, scores = pickle.load(inp)


Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_TRAIN)
mean_abs_signal = np.mean(np.abs(np.array(Ytrain[1])))

plt.figure()
for key in scores.keys():
    plt.plot(noise_levels, scores[key], label="N=" + str(key))
plt.legend()