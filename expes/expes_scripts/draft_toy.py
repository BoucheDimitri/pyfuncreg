import os
import pickle

rec_path = os.getcwd() + "/outputs/toy_kam_kertuning"

with open(rec_path + "/full.pkl", "rb") as inp:
    best_conf, best_result, score_test = pickle.load(inp)