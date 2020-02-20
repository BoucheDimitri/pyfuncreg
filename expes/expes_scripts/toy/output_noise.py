import numpy as np
import os
import sys
import pickle
import pathlib

# Execution path
exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
path = str(exec_path.parent.parent.parent)
sys.path.append(path)

# Local imports
from data import degradation
from functional_data import basis
from functional_regressors import kernels
from data import toy_data_spline
from solvers import first_order
from functional_regressors import kernel_projection_learning as kproj_learning
from misc import model_eval

# ############################### Config ###############################################################################
# Record config
OUTPUT_FOLDER = "output_noise"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "output_noise"

# ############################### Experiment parameters ################################################################
N_TRAIN = 500
KER_SIGMA = 20
# REGU_GRID = np.geomspace(1e-7, 1e-4, 100)
REGU_GRID = [1e-7, 1e-6]
NOISE_INPUT = 0.07
# NOISE_OUTPUT = np.linspace(0, 1.5, 50)
NOISE_OUTPUT = [0.1, 0.2]
SEED_INPUT = 768
SEED_OUTPUT = 456
N_FOLDS = 5

# ############################### Optim parameters #####################################################################
MAXIT = 1000
TOL = 1e-8


if __name__ == '__main__':

    # ############################# Create folder for recording ########################################################
    try:
        os.mkdir(path + "/outputs")
    except FileExistsError:
        pass
    try:
        os.mkdir(path + "/outputs/" + OUTPUT_FOLDER)
    except FileExistsError:
        pass
    rec_path = path + "/outputs/" + OUTPUT_FOLDER

    # ############################## Generate toy dataset ##############################################################
    Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(N_TRAIN)
    # Add input noise
    Xtrain = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, SEED_INPUT)

    # ############################## Build regressor ###################################################################
    # Spline dict
    locs_bounds = np.array([toy_data_spline.BOUNDS_FREQS[0], toy_data_spline.BOUNDS_FREQS[1]])
    domain = toy_data_spline.DOM_OUTPUT
    func_dict = basis.BSplineUniscaleBasis(domain, toy_data_spline.BOUNDS_FREQS[-1],
                                           locs_bounds, width=toy_data_spline.WIDTH)
    # Scalar kernel
    gauss_ker = kernels.GaussianScalarKernel(KER_SIGMA, normalize=False)
    # Operator valued kernel matrix
    B = np.eye(func_dict.n_basis)
    # Solver
    bfgs = first_order.ScipySolver(maxit=MAXIT, tol=TOL, method="L-BFGS-B")
    regressors = [kproj_learning.KPLApprox(gauss_ker, B, func_dict, reg, bfgs) for reg in REGU_GRID]

    # ############################## Run experiment ####################################################################
    cross_val = model_eval.CrossValidationEval(n_folds=N_FOLDS)
    scores_list = []
    count = 0
    for noise in NOISE_OUTPUT:
        Ytrain_deg = degradation.add_noise_outputs(Ytrain, noise, SEED_OUTPUT)
        regu_results = []
        for i in range(len(REGU_GRID)):
            regu_results.append(cross_val(regressors[i], Xtrain, Ytrain_deg))
        best_regressor = regressors[np.argmin(regu_results)]
        best_regressor.fit(Xtrain, Ytrain_deg)
        score_test = model_eval.mean_squared_error(best_regressor.predict_evaluate(Xtest, Ytest[0][0]), Ytest[1])
        scores_list.append(score_test)
        print("Noise level number " + str(count) + " tested. Remaining: " + str(len(NOISE_OUTPUT) - count))
        count += 1
    # Save result
    with open(rec_path + "/" + EXPE_NAME + ".pkl", "wb") as inp:
        pickle.dump((NOISE_OUTPUT, scores_list), inp, pickle.HIGHEST_PROTOCOL)