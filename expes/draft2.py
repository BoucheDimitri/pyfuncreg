import pickle
import os
import numpy as np

path = os.getcwd() + "/outputs/"
KEYS = ["LA"]

folders_speech = ["speech_kpl_rffs_multi"]

with open(path + "speech_kpl_rffs_multi/0_LA.pkl", "rb") as inp:
    best_config_3be, best_result_3be, score_test_3be = pickle.load(inp)

