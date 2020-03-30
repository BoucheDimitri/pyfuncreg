import pickle
import os

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_15-43/outputs/toy_correlated_data3/"
# path = "/home/dimitri/Desktop/Telecom/nonlinear_functional_regressions/outputs/toy_correlated/"
with open(path + "full.pkl", "rb") as inp:
    n_samples, scores_test, scores_test_corr = pickle.load(inp)

results_list_corr = []
for n in n_samples:
    with open(path + str(n) + "_corr.pkl", "rb") as inp:
        results_list_corr.append(pickle.load(inp))

results_list = []
for n in n_samples:
    with open(path + str(n) + ".pkl", "rb") as inp:
        results_list.append(pickle.load(inp))