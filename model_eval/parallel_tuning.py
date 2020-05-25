import numpy as np
import multiprocessing
import time
import pickle
import psutil

from model_eval import cross_validation
from model_eval import metrics
from functional_data import discrete_functional_data as disc_fd


def check_cpu_availability(min_nprocs=32, timeout_sleep=3, n_timeout=0, cpu_avail_thresh=90):
    """
    Allocate CPUs for execution based on a minimum number of available CPUs for `n_procs`, if the number of
    CPUs available are inferior to that number, the function can be set to wait and re-try several times.

    Parameters
    ----------
    min_nprocs: int
        The number of CPUs desired for execution
    timeout_sleep: float
        The duration of the waiting periods if less that `min_nprocs` CPUs are available
    n_timeout: int
        Number of waiting periods allowed
    cpu_avail_thresh:
        Percentage of occupation under which a CPU is considered available

    Returns
    -------
    int:
        The number of available CPUs for execution
    """
    cpu_occup = np.array(psutil.cpu_percent(percpu=True))
    return len(cpu_occup)
    # TODO: Shortcut chanmé attention
    # n_procs = (cpu_occup[cpu_occup < cpu_avail_thresh]).shape[0]
    # timeout_count = 0
    # while n_procs < min_nprocs and timeout_count < n_timeout:
    #     time.sleep(timeout_sleep)
    #     cpu_occup = np.array(psutil.cpu_percent(percpu=True))
    #     n_procs = (cpu_occup[cpu_occup < 90]).shape[0]
    #     timeout_count += 1
    # if n_procs < min_nprocs:
    #     raise ResourceWarning("The minimum number of CPUs required could not be allocated")
    # return n_procs


def run_cross_vals_batch(regs_split, cross_val, Xfit, Yfit, Xpred=None, Ypred=None):
    with multiprocessing.Pool(processes=len(regs_split)) as pool:
        multiple_results = [pool.apply_async(cross_val, (regs_split[j], Xfit, Yfit, Xpred, Ypred,))
                            for j in range(len(regs_split))]
        return [res.get() for res in multiple_results]


def record_up_to_current_batch(rec_path, key, results, configs, batch_no, n_batches, n_procs):
    if key is None or "":
        key_addup = ""
    else:
        key_addup = "_" + key
    if rec_path is not None:
        if configs is not None:
            with open(rec_path + "/batch_no" + str(batch_no)
                      + "out_of" + str(n_batches) + key_addup + ".pkl", "wb") as outp:
                pickle.dump((configs[:(batch_no + 1) * n_procs], results[:(batch_no + 1) * n_procs]), outp,
                            pickle.HIGHEST_PROTOCOL)
        else:
            with open(rec_path + "/batch_no" + str(batch_no)
                      + "out_of" + str(n_batches) + key_addup + ".pkl", "wb") as outp:
                pickle.dump(results[:(batch_no + 1) * n_procs], outp, pickle.HIGHEST_PROTOCOL)


def parallel_cross_vals(regs, cross_val, Xfit, Yfit, Xpred, Ypred, n_procs, rec_path=None, key=None, configs=None):
    """
    Cross validate the regressors in reg in parallel.

    Parameters
    ----------
    regs : list or tuple
        The regressors to cross-validate, must implement the methods **fit** and **predict_evaluate_diff_locs**
    Xfit :
        The input data used for fitting
    Yfit :
        The output data used for fitting
    Xpred :
        The input data used for testing
    Ypred :
        The output data used for testing
    cross_val : model_eval.cross_validation.KfoldsCrossVal
        The cross validation to run
    n_procs : int
        Number of processor to use for parallelization
    rec_path : str, optional
        Path for recording incrementally the batches of results
    key : str, optional
        String added to the file name of the batch if `rec_path` is not None
    configs : dict, optional
        The dictionaries of configuration corresponding to the regressors in `regs`, if `rec_path` is not None,
        the corresponding dictionaries are saved along with the results

    Returns
    -------
    list
        The list of cross-validation scores
    """
    # Number of batches
    n_batches = len(regs) // n_procs
    results = []
    if n_batches == 0:
        results += run_cross_vals_batch(regs, cross_val, Xfit, Yfit, Xpred, Ypred)
    else:
        # Execute batches sequentially
        for i in range(n_batches + 1):
            if i < n_batches:
                # Create batch
                regs_split = regs[i * n_procs: (i + 1) * n_procs]
                # Execute batch
                results += run_cross_vals_batch(regs_split, cross_val, Xfit, Yfit, Xpred, Ypred)
                # Print progress
                print("Process batch number " + str(i) + " finished. Remaining: " + str(n_batches - i - 1))
                # Record results up to current batch
                record_up_to_current_batch(rec_path, key, results, configs, i, n_batches, n_procs)
            else:
                regs_split = regs[i * n_procs:]
                if len(regs_split) > 0:
                    results += run_cross_vals_batch(regs_split, cross_val, Xfit, Yfit, Xpred, Ypred)
    return results


def parallel_tuning(regs, Xfit_train, Yfit_train, Xtest, Ytest, Xpred_train=None, Ypred_train=None,
                    rec_path=None, key=None, configs=None, input_indexing="discrete_general",
                    output_indexing="discrete_general", n_folds=5, n_procs=None, min_nprocs=4,
                    timeout_sleep=3, n_timeout=10, cpu_avail_thresh=90):
    """
    Tuning of the regressors in parallel by cross-validation, selecting the best and fitting it on the train set,
    its score on test set is then computed.

    Parameters
    ----------
    regs: list or tuple
        list or tuple of functional_regressors.functional_regressor.FunctionalRegressor
    Xtrain:
        The training input data in the format corresponding to `cv_mode`
    Ytrain: list of tuple, len = 2
        The training output data,`Ytrain` = (Ytrain_locs, Ytrain_obs), Ytrain_locs and Ytrain_obs of len = n_samples_train
        and for 1 <= i <= n_samples, Ytrain_locs[i] and Ytrain_obs[i] have shape = [n_observations_i, 1]
    Xtest:
        The testing input data in the format corresponding to `cv_mode`
    Ytest: list or tuple, len = 2
        The testing output data,`Ytest` = (Ytest_locs, Ytest_obs), Ytest_locs and Ytest_obs of len = n_samples_test
        and for 1 <= i <= n_samples, Ytest_locs[i] and Ytest_obs[i] have shape = [n_observations_i, 1]
    rec_path: str, optional
        Path for recording incrementally the batches of results
    key: str, optional
        String added to the file name of the batch if `rec_path` is not None
    configs: dict, optional
        The dictionaries of configuration corresponding to the regressors in `regs`, if `rec_path` is not None,
        the corresponding dictionaries are saved along with the results
    input_data_format: {"vector", "discrete_general", discrete_samelocs_regular_1d"}
        The format of the input data
    output_data_format: {"vector", "discrete_general", discrete_samelocs_regular_1d"}
        The format of the output data
    n_folds: int
        Number of folds for cross-validation
    n_procs: int
        User defined number of CPUs, overides the other CPU execution parameters
    min_nprocs: int
        The number of CPUs desired for execution
    timeout_sleep: float
        The duration of the waiting periods if less that `min_nprocs` CPUs are available
    n_timeout: int
        Number of waiting periods allowed
    cpu_avail_thresh:
        Percentage of occupation under which a CPU is considered available

    Returns
    -------
    tuple
        Either (best_config, crossval_score_train, score_test) if `configs` is not None or
        (crossval_score_train, score_test) otherwise.
    """
    # Number of processors for execution
    if n_procs is None:
        n_procs = check_cpu_availability(min_nprocs=min_nprocs, timeout_sleep=timeout_sleep,
                                         n_timeout=n_timeout, cpu_avail_thresh=cpu_avail_thresh)
    # Instantiate cross validation
    cross_val = cross_validation.KfoldsCrossVal(n_folds=n_folds, input_indexing=input_indexing,
                                                output_indexing=output_indexing)
    # Run cross-validations in parallel for the regressors in regs
    results = parallel_cross_vals(regs, cross_val, Xfit_train, Yfit_train, Xpred_train, Ypred_train,
                                  n_procs, rec_path, key, configs)
    # Select regressor corresponding to best cross-validation score
    best_ind = int(np.argmin(results))
    best_reg = regs[best_ind]
    # Fit best regressor on full training data
    best_reg.fit(Xfit_train, Yfit_train)
    # Evaluate its performance on test set
    # Put in discrete_general form for testing
    Ytest_dg = disc_fd.to_discrete_general(*Ytest)
    preds = best_reg.predict_evaluate_diff_locs(Xtest, Ytest_dg[0])
    score_test = metrics.mse(preds, Ytest_dg[1])
    # Return the results
    if configs is not None:
        return configs[best_ind], results[best_ind], score_test
    else:
        return results[best_ind], score_test

