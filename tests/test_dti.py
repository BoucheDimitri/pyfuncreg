import numpy as np
import os

from expes.expes_tools import generate_expes
from data import loading, processing
from misc import model_eval

SHUFFLE_SEED = 784
N_TRAIN = 70

# ############################ Load DTI data ###########################################################################
cca, rcst = loading.load_dti(os.getcwd() + "/data/dataDTI/", shuffle_seed=SHUFFLE_SEED)

Xtrain0, Ytrain0, Xtest0, Ytest0 = processing.process_dti_bis(cca.copy(), rcst.copy(), n_train=N_TRAIN,
                                                              normalize01=True, interpout=False, pad_train=True,
                                                          pad_width=((0, 0), (0, 0)), pad_mode="symmetric")

Xtrain1, Ytrain1, Xtest1, Ytest1 = processing.process_dti_dataset(cca.copy(), rcst.copy(), n_train=N_TRAIN,
                                                                  normalize01=True, interpout=False,
                                                                  pad_train_output=True, pad_train_input=True,
                                                                  pad_width_output=((0, 0), (0, 0)),
                                                                  pad_mode_output="symmetric",
                                                                  pad_train_input=True,
                                                                  pad_width_input=((0, 0), (0, 0)),
                                                                  pad_mode_input="symmetric")



Xtrain, Ytrain, Xtest, Ytest = Xtrain1, Ytrain1, Xtest1, Ytest1
Xtrain, Ytrain, Xtest, Ytest = Xtrain0, Ytrain0, Xtest0, Ytest0
# ############################# Kernel additive model (KAM) ############################################################
# Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
#                                                               interpout=False, pad_train_input=False,
#                                                               pad_train_output=False)

# Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_bis(cca.copy(), rcst.copy(), n_train=N_TRAIN,
#                                                           normalize01=True, interpout=False, pad_train=True,
#                                                           pad_width=((0, 0), (0, 0)), pad_mode="symmetric")

domain_out = np.array([[0, 1]])
nevals_in = 100
nevals_out = 60
nevals_fpca = 500

dict_test = {'regu': 0.007564633275546291, 'kx': 0.1, 'ky': 0.05, 'keval': 0.1, 'nfpca': 30}
reg = generate_expes.create_kam_dti(dict_test, nevals_in, nevals_out, domain_out, domain_out, nevals_fpca)
reg.fit(Xtrain, Ytrain)
len_test = len(Xtest[0])
test_pred = [reg.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len(Xtest[0]))]
score = model_eval.mean_squared_error(test_pred, [Ytest[1][i] for i in range(len_test)])


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
pad_width_output = ((0, 0), (55, 55))
pad_width_input = ((0, 0), (0, 0))
domain_out_pad = np.array([[-pad_width_output[1][0] / 55, 1 + pad_width_output[1][0] / 55]])
# Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca, rcst, n_train=N_TRAIN, normalize01=True,
#                                                               interpout=False, pad_train_input=False,
#                                                               pad_train_output=True, pad_width_output=pad_width,
#                                                               pad_mode_output="symmetric")
# Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_bis(cca, rcst, n_train=N_TRAIN, normalize01=True,
#                                                           interpout=False, pad_train=True, pad_width=pad_width,
#                                                           pad_mode="symmetric")
Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca.copy(), rcst.copy(), normalize01=True,
                                                              pad_mode_input="symmetric",
                                                              pad_width_input=pad_width_input,
                                                              pad_mode_output="symmetric",
                                                              pad_width_output=pad_width_output)
Xtrain = np.array(Xtrain[1]).squeeze()
Xtest = np.array(Xtest[1]).squeeze()
dict_test = {"regu": 0.009236708571873866, "ker_sigma": 0.9, "pywt_name": "db", "init_dilat": 1, "dilat": 2, "translat": 1,
             "moments": 2, "n_dilat": 5, "center_outputs": True,
             "penalize_freqs": 1.0, "add_constant": True}
reg = generate_expes.create_kpl_dti(dict_test, domain_out, domain_out_pad, pad_width_output)
reg.fit(Xtrain, Ytrain)
test_pred = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
score = model_eval.mean_squared_error(test_pred, Ytest[1])

# ############################# Triple basis estimator (3BE) ###########################################################
domain_out = np.array([[0, 1]])
nrffs = 300
rffs_seed = 567
pad_width = ((0, 0), (0, 0))

Xtrain, Ytrain, Xtest, Ytest = processing.process_dti_dataset(cca.copy(), rcst.copy(), normalize01=True,
                                                              pad_mode_output="symmetric", pad_width_output=pad_width)

dict_test = {"regu": 1, "ker_sigma": 20, "max_freq_in": 25,
             "max_freq_out": 5, "center_outputs": True}


reg = generate_expes.create_3be_dti(dict_test, domain_out, domain_out, nrffs, rffs_seed, pad_width)
reg.fit(Xtrain, Ytrain)
test_pred = [reg.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len(Xtest[0]))]
score = model_eval.mean_squared_error(test_pred, [Ytest[1][i] for i in range(len_test)])