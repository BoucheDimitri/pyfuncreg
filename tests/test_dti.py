import numpy as np
import os

from expes.expes_tools import generate_expes
from data import loading, processing

SHUFFLE_SEED = 784
N_TRAIN = 70

# ############################ Load DTI data ###########################################################################
cca, rcst = loading.load_dti(os.getcwd() + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)


# ############################# Kernel additive model (KAM) ############################################################
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
                                                              interpout=False, pad_train_input=False,
                                                              pad_train_output=False)
domain_out = np.array([[0, 1]])
nevals_in = 100
nevals_out = 60
nevals_fpca = 500

dict_test = {"regu": 1e-3, "kx": 0.05, "ky": 0.05, "keval": 0.03, "nfpca": 30}
reg = generate_expes.create_kam_dti(dict_test, nevals_in, nevals_out, domain_out, domain_out, nevals_fpca)
reg.fit(Xtrain, Ytrain)
test_pred = reg.predict_evaluate(Xtest, Ytest[0][0])


# ############################# Kernel estimator (KE) ##################################################################
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
                                                              interpout=False, pad_train_input=False,
                                                              pad_train_output=False)
Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

dict_test = {"window": 0.1}

reg = generate_expes.create_ke_dti(dict_test)
reg.fit(Xtrain, Ytrain)
test_pred = reg.predict_evaluate(Xtest, Ytest[0][0])


# ############################# Functional kernel ridge regression (FKR) ###############################################
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
                                                              interpout=False, pad_train_input=False,
                                                              pad_train_output=False)
Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

approx_locs = np.linspace(0, 1, 200)
dict_test = {"regu": 1e-6, "ker_sigma": 0.9, "ky": 0.1, "center_outputs": True}

reg = generate_expes.create_fkr_dti(dict_test, approx_locs)
reg.fit(Xtrain, Ytrain)
test_pred = reg.predict_evaluate(Xtest, Ytest[0][0])


# ############################# Kernel-based projection learning #######################################################
domain_out = np.array([[0, 1]])
pad_width = ((0, 0), (165, 165))
domain_out_pad = np.array([[-pad_width[1][0] / 55, 1 + pad_width[1][0] / 55]])
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
                                                              interpout=False, pad_train_input=False,
                                                              pad_train_output=True, pad_width_output=pad_width,
                                                              pad_mode_output="symmetric")
Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()

dict_test = {"regu": 1e-3, "ker_sigma": 0.9, "pywt_name": "db", "init_dilat": 1, "dilat": 2, "translat": 1,
             "moments": 2, "n_dilat": 5, "center_outputs": True,
             "penalize_freqs": 1, "add_constant": True}
reg = generate_expes.create_kpl_dti(dict_test, domain_out, domain_out_pad, pad_width)
reg.fit(Xtrain, Ytrain)
test_pred = reg.predict_evaluate(Xtest, Ytest[0][0])