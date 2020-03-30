import pickle
import os

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_30-03-2020_11-02/outputs/toy_correlated_data2/"
# path = "/home/dimitri/Desktop/Telecom/nonlinear_functional_regressions/outputs/toy_correlated/"
with open(path + "full.pkl", "rb") as inp:
    n_samples, scores_test, scores_test_corr = pickle.load(inp)