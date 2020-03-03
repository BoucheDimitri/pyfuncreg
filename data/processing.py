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
