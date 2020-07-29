from data import processing
from data import loading
from data import degradation
import os
import numpy as np
import matplotlib.pyplot as plt

X, Y = loading.load_raw_speech_dataset(os.getcwd() + "/data/dataspeech/raw/")
key = "VEL"

Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
        X, Y, shuffle_seed=543, n_train=300, normalize_domain=True, normalize_values=True)
Ytrain_ext, Ytrain, Ytest_ext, Ytest = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]

random_state = np.random.RandomState(453)
missing_rate = 0.7
Ytrain, Ytrain_ext = degradation.downsample_output_nan_ext(Ytrain, Ytrain_ext, missing_rate, random_state)

plt.figure()
plt.scatter(Ytrain[0][0], Ytrain[1][0])

plt.figure()
plt.scatter(Ytrain_ext[0][0], Ytrain_ext[1][0])