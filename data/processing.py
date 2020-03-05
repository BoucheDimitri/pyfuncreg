import numpy as np


def process_dti(cca, rcst, n_train=70, normalize01=True, interp_input=True):
    # Add sampling locations
    locs_cca = np.arange(0, cca.shape[1])
    locs_rcst = np.arange(0, rcst.shape[1])
    # Use interpolation for filling NaNs in the input data if relevant
    if interp_input:
        for i in range(len(cca)):
            cca[i] = np.interp(locs_cca,
                               locs_cca[np.argwhere(~ np.isnan(cca[i])).squeeze()],
                               cca[i][np.argwhere(~ np.isnan(cca[i])).squeeze()])
    # Normalize locations to [0, 1] if relevant
    if normalize01:
        locs_cca = (1 / cca.shape[1]) * np.arange(0, cca.shape[1])
        locs_rcst = (1 / rcst.shape[1]) * np.arange(0, rcst.shape[1])
    return (locs_cca, cca[:n_train]), (locs_rcst, rcst[:n_train]), \
           (locs_cca, cca[n_train:]), (locs_rcst, rcst[n_train:])


# TODO: COMMENT SE DEBROUILLER DU FAIT QU'ICI ON NE VA PAS UTILISER LE MEME TYPE DE DATA POUR LE FITTING ET POUR LE
# TODO: ET POUR LE TESTING ? ATTENTION DU COUP LA FONCTION CI-DESSOUS EST FAUSSE

def process_speech_1VT(Xtrain, Ytrain, Xtest, Ytest, vocal_tract="LA", normalize_output=True):
    Ytrain_sub, Ytest_sub = Ytrain[vocal_tract], Ytest[vocal_tract]
    if normalize_output:
        m = np.min(np.array(Ytrain[vocal_tract][1]))
        M = np.max(np.array(Ytrain[vocal_tract][1]))
        a = 2 / (M - m)
        b = 1 - a * M
    else:
        a = 1
        b = 0
    return Xtrain, (Ytrain_sub[0][0], a * np.array(Ytrain_sub[1]) + b), Xtest, (Ytest_sub[0][0], a * np.array(Ytest_sub[1]) + b)

# def process_speech(Xtrain, Ytrain, Xtest, Ytest, normalize_output=True):
#     if normalize_output:
#         Ytrain_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
#                          "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
#         Ytest_normalized = {"LP": [[], []], "LA": [[], []], "TBCL": [[], []], "TBCD": [[], []],
#                          "VEL": [[], []], "GLO": [[], []], "TTCL": [[], []], "TTCD": [[], []]}
#         for key in Ytrain.keys():
#             m = np.min(np.array(Ytrain[key][1]))
#             M = np.max(np.array(Ytrain[key][1]))
#             a = 2 / (M - m)
#             b = 1 - a * M
#             for j in range(len(Ytrain[key][0])):
#                 Ytrain_normalized[key][1].append(a * Ytrain[key][1][j] + b)
#                 Ytrain_normalized[key][0].append(Ytrain[key][0][j])
#             for j in range(len(Ytest[key][1])):
#                 Ytest_normalized[key][1].append(a * Ytest[key][1][j] + b)
#                 Ytest_normalized[key][0].append(Ytest[key][0][j])
#     else:
#         Ytrain_normalized = Ytrain
#         Ytest_normalized = Ytest
#     return Xtrain, Ytrain_normalized[0], Xtest, Ytest_normalized