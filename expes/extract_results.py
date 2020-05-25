import pickle
import os
import numpy as np


# ################################## DTI ###############################################################################

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

path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_16-04-2020_11-33/outputs/"

# folders_dti = ["dti_3be_multi", "dti_fkrr_multi", "dti_kam_multi", "dti_kpl_multi", "dti_ke_multi"]
folders_dti = ["dti_3be_multi_wavs", "dti_fkrr_multi", "dti_kpl_multi3", "dti_ke_multi", "dti_kam_multi"]


with open(path + "dti_kpl_multi2" + "/9.pkl", "rb") as inp:
    _1, _2, score_test = pickle.load(inp)

with open(path + "dti_3be_multi_wavs" + "/9.pkl", "rb") as inp:
    _1, _2, score_test = pickle.load(inp)

with open(path + "dti_kam_multi" + "/3.pkl", "rb") as inp:
    _1, _2, score_test = pickle.load(inp)

for folder in folders_dti:
    print(folder)
    m, s = mean_variance_result_dti(path + folder)
    print("mean:" + str(m))
    print("std: " + str(s))



# ############################ SPEECH ##################################################################################
path = "/home/dimitri/Desktop/Telecom/Outputs/all_outputs_19-05-2020_08-28/outputs/"
KEYS = ("LP", "LA", "TBCL", "TBCD", "VEL", "GLO", "TTCL", "TTCD")
# KEYS = ("LP", "LA", "TBCL", "TBCD")

# folders_speech = ["speech_3be_fourier", "speech_kpl_rffsmax", "speech_ke_multi", "speech_fkrr_multi"]
# folders_speech = ["speech_3be_multi_max", "speech_kpl_rffs300", "speech_ke_multi", "speech_fkrr_multi"]
folders_speech = ["speech_kpl_rffs200", "speech_kpl_rffs100"]

# with open(path + "speech_3be_multi/9_LP.pkl", "rb") as inp:
#     best_config_3be, best_result_3be, score_test_3be = pickle.load(inp)
#
with open(path + "speech_fkrr_multi/9_LP.pkl", "rb") as inp:
    best_config_200, best_result_200, score_test_200 = pickle.load(inp)



def mean_variance_result_speech(path, key):
    with open(path + "/9_" + key + ".pkl", "rb") as inp:
        _1, _2, score_test = pickle.load(inp)
    return np.mean(score_test), np.std(score_test)

for key in KEYS:
    print(key)
    for folder in folders_speech:
        print(folder)
        m, s = mean_variance_result_speech(path + folder, key)
        print("mean:" + str(m))
        print("std: " + str(s))
    print(" ")
