import numpy as np
import os

from expes.expes_tools import generate_expes
from data import loading, processing
from misc import model_eval

SHUFFLE_SEED = 784
N_TRAIN = 300

# ##################################### Load the data and process it ###################################################
Xraw, Yraw = loading.load_speech_dataset_bis(os.getcwd() + "/data/dataspeech/raw/")
Xtrain, Ytrain, Xtest, Ytest = processing.process_speech_dataset(Xraw, Yraw, shuffle_seed=SHUFFLE_SEED, n_train=N_TRAIN)
Ytrain_sub, Ytest_sub = Ytrain["LA"], Ytest["LA"]


# ############################# Kernel estimator (KE) ##################################################################
dict_test = {"ker_sigma": 1, "center_output": True}
reg = generate_expes.create_ke_speech(dict_test)
reg.fit(Xtrain, Ytrain_sub)
test_pred = reg.predict_evaluate(Xtest, Ytest_sub[0][0])


# ############################# Functional kernel ridge (FKR) ##########################################################
approx_locs = np.linspace(0, 1, 300)
dict_test = {"regu": 1e-6, "ker_sigma": 0.9, "ky": 0.1, "center_outputs": True}

reg = generate_expes.create_fkr_speech(dict_test, approx_locs)
reg.fit(Xtrain, Ytrain_sub)
test_pred = reg.predict_evaluate(Xtest, Ytest_sub[0][0])


# ############################# Kernel-based projection learning (KPL) #################################################
nevals_fpca = 300
dict_test = {"regu": 1e-4, "ker_sigma": 1,
             "penalize_eigvals": [0], "n_fpca": 30, "penalize_pow": [1], "center_output": True}

reg = generate_expes.create_kpl_speech(dict_test, nevals_fpca)
reg.fit(Xtrain, Ytrain_sub)
test_pred = reg.predict_evaluate(Xtest, Ytest_sub[0][0])


# ############################# Triple-basis estimator (3BE) ###########################################################
nevals_fpca = 300
dict_test = {"regu": 1e-4, "ker_sigma": 1, "nfpca": 30, "center_output": True}
reg = generate_expes.create_3be_speech(dict_test, nevals_fpca)
reg.fit(Xtrain, Ytrain_sub)
test_pred = reg.predict_evaluate(Xtest, Ytest_sub[0][0])