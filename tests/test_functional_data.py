import os
import numpy as np
import importlib

from data import loading
from functional_data import discrete_functional_data

importlib.reload(discrete_functional_data)

Xtrain, Ytrain, Xtest, Ytest = loading.load_dti(os.getcwd() + "/data/dataDTI/")

rcst_test = discrete_functional_data.DiscreteSamelocsRegular1D(np.arange(0, 55, 1), Ytrain)

Ylocs, Yobs = rcst_test.discrete_general()

rcst_test_ext = rcst_test.extended_version(repeats=(2, 2))

Ylocs, Yobs = rcst_test_ext.discrete_general()