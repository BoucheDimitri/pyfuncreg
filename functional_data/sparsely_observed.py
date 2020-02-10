import numpy as np


def mean_missing(Ylocs, Yevals):
    """
    Compute mean from sparse functional sample ignoring missing values
    """
    n = len(Ylocs)
    lens = np.array([len(Ylocs[i]) for i in range(n)])
    full_locs = Ylocs[np.argmax(lens)]
    dimy = np.max(lens)
    temp = np.full((n, dimy), np.nan)
    for i in range(n):
        n_missing = 0
        for j in range(dimy):
            if full_locs[j] in Ylocs[i]:
                temp[i, j] = Yevals[i][j - n_missing]
            else:
                n_missing += 1
    return full_locs, np.nanmean(temp, axis=0)


def substract_missing(full_locs, Ymean, Ylocs, Yevals):
    """
    Substract mean from sample with missing values ignoring those
    """
    Yevals_centered = []
    n = len(Ylocs)
    dimy = full_locs.shape[0]
    for i in range(n):
        n_missing = 0
        ycopy = Yevals[i].copy()
        for j in range(dimy):
            if full_locs[j] in Ylocs[i]:
                ycopy[j - n_missing] -= Ymean[j]
            else:
                n_missing += 1
        Yevals_centered.append(ycopy)
    return Ylocs, Yevals_centered