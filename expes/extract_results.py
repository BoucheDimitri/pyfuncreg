import pickle
import os
import numpy as np


# def mean_variance_result_dti(path):
#     scores_test = []
#     for f in os.listdir(path):
#         with open(path + "/" + f, "rb") as inp:
#             _1, _2, score_test = pickle.load(inp)
#         scores_test.append(score_test)
#     return scores_test
#     return np.mean(scores_test), np.std(scores_test)

def mean_variance_result_dti(path):
    with open(path + "/9.pkl", "rb") as inp:
        _1, _2, score_test = pickle.load(inp)
    return np.mean(score_test), np.std(score_test)

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_06-04-2020_09-42/outputs/"

# folders_dti = ["dti_3be_multi", "dti_fkrr_multi", "dti_kam_multi", "dti_kpl_multi", "dti_ke_multi"]
folders_dti = ["dti_3be_multi", "dti_fkrr_multi", "dti_kpl_multi", "dti_ke_multi"]


with open(path + "dti_kpl_multi" + "/9.pkl", "rb") as inp:
    _1, _2, score_test = pickle.load(inp)

with open(path + "dti_3be_multi" + "/9.pkl", "rb") as inp:
    _1, _2, score_test = pickle.load(inp)

for folder in folders_dti:
    print(folder)
    m, s = mean_variance_result_dti(path + folder)
    print("mean:" + str(m))
    print("std: " + str(s))