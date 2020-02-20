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
OUTPUT_FOLDER = "output_missing"
REC_PATH = path + "/outputs/" + OUTPUT_FOLDER
EXPE_NAME = "output_missing"

# ############################### Experiment parameters ################################################################
N_TRAIN = 500
KER_SIGMA = 20
REGU_GRID = np.geomspace(1e-7, 1e-4, 100)
NOISE_INPUT = 0.07
NOISE_OUTPUT = 0.02
SEED_INPUT = 768
SEED_OUTPUT = 456
NSAMPLES_LIST = [20, 50, 100, 250, 500]
MISSING_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
SEED_MISSING = 560
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
    products_list = []
    for n_samples in NSAMPLES_LIST:
        Xtrain, Ytrain, Xtest, Ytest = toy_data_spline.get_toy_data(n_samples)
        for deg in MISSING_LEVELS:
            Xtrain_deg = degradation.add_noise_inputs(Xtrain, NOISE_INPUT, SEED_INPUT)
            Ytrain_deg = degradation.add_noise_outputs(Ytrain, NOISE_OUTPUT, SEED_OUTPUT)
            Ytrain_deg = degradation.downsample_output(Ytrain_deg, deg, SEED_MISSING)
            regu_results = []
            for i in range(len(REGU_GRID)):
                regu_results.append(cross_val(regressors[i], Xtrain_deg, Ytrain_deg))
                best_regressor = regressors[np.argmin(regu_results)]
                best_regressor.fit(Xtrain_deg, Ytrain_deg)
                score_test = model_eval.mean_squared_error(best_regressor.predict_evaluate(Xtest, Ytest[0][0]),
                                                           Ytest[1])
                scores_list.append(score_test)
                products_list.append((n_samples, deg))
        print("Finished for N=" + str(n_samples))
    # Save result
    with open(rec_path + "/" + EXPE_NAME + ".pkl", "wb") as inp:
        pickle.dump((products_list, scores_list), inp, pickle.HIGHEST_PROTOCOL)
