import pickle
import matplotlib.pyplot as plt
import os

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/output_missing2/"
# path = "/home/dimitri/Desktop/Telecom/nonlinear_functional_regressions/outputs/toy_correlated/"
with open(path + "full.pkl", "rb") as inp:
    missing_levels, scores = pickle.load(inp)

plt.figure()
for key in scores.keys():
    plt.plot(missing_levels, scores[key], label="N=" + str(key))
plt.legend()