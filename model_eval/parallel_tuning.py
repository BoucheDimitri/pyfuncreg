import numpy as np
import multiprocessing
import time
import pickle
import psutil

from model_eval import cross_validation
from model_eval import metrics


def check_cpu_availability(min_nprocs=4, timeout_sleep=3, n_timeout=0, cpu_avail_thresh=30):
    """
    Allocate CPUs for execution based on a minimum number of available CPUs ` for n_procs`, if the number of
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
    n_procs = (cpu_occup[cpu_occup < cpu_avail_thresh]).shape[0]
    timeout_count = 0
    while n_procs < min_nprocs and timeout_count < n_timeout:
        time.sleep(timeout_sleep)
        cpu_occup = np.array(psutil.cpu_percent(percpu=True))
        n_procs = (cpu_occup[cpu_occup < 90]).shape[0]
        timeout_count += 1
    if n_procs < min_nprocs:
        raise ResourceWarning("The minimum number of CPUs required could not be allocated")
    return n_procs


def parallel_cross_vals(regs, Xtrain, Ytrain, cross_val, n_procs, rec_path=None, key=None, configs=None):
    """
    Cross validate the regressors in reg in parallel.

    Parameters
    ----------
    regs: list or tuple
        The regressors to cross-validate, must implement the methods **fit** and **predict_evaluate_diff_locs**
    Xtrain:
        The input data corresponding to the mode in `cross_val`
    Ytrain:
        The output data
    cross_val: model_eval.cross_validation.KfoldsCrossVal
        The cross validation to run
    n_procs: int
        Number of processor to use for parallelization
    rec_path: str, optional
        Path for recording incrementally the batches of results
    key: str, optional
        String added to the file name of the batch if `rec_path` is not None
    configs: dict, optional
        The dictionaries of configuration corresponding to the regressors in `regs`, if `rec_path` is not None,
        the corresponding dictionaries are saved along with the results

    Returns
    -------
    list
        The list of cross-validation scores
    """
    # Number of experiments
    nexpes = len(regs)
    # Number of experiments batch
    nsplits = len(regs) // n_procs
    results = []
    if nsplits == 0:
        with multiprocessing.Pool(processes=nexpes) as pool:
            multiple_results = [pool.apply_async(cross_val, (regs[i], Xtrain, Ytrain, )) for i in range(nexpes)]
            results += [res.get() for res in multiple_results]
    else:
        count = 0
        for i in range(nsplits + 1):
            if count < nsplits:
                regs_split = regs[i * n_procs: (i + 1) * n_procs]
                with multiprocessing.Pool(processes=n_procs) as pool:
                    multiple_results = [pool.apply_async(cross_val, (regs_split[j], Xtrain, Ytrain, ))
                                        for j in range(len(regs_split))]
                    results += [res.get() for res in multiple_results]
                print("Process batch number " + str(count) + " finished. Remaining: " + str(nsplits - count - 1))
                if rec_path is not None:
                    with open(rec_path + "/batch_no" + str(count) + "out_of"
                              + str(nsplits) + "_" + key + ".pkl", "wb") as outp:
                        if configs is not None:
                            pickle.dump((configs[:(i + 1) * n_procs], results[:(i + 1) * n_procs]), outp,
                                        pickle.HIGHEST_PROTOCOL)
                        else:
                            pickle.dump(results[:(i + 1) * n_procs], outp,
                                        pickle.HIGHEST_PROTOCOL)
                count += 1
            else:
                regs_split = regs[i * n_procs:]
                if len(regs_split) > 0:
                    with multiprocessing.Pool(processes=len(regs_split)) as pool:
                        multiple_results = [pool.apply_async(cross_val, (regs_split[j], Xtrain, Ytrain, ))
                                            for j in range(len(regs_split))]
                        results += [res.get() for res in multiple_results]
    return results


def parallel_tuning(regs, Xtrain, Ytrain, Xtest, Ytest, rec_path=None, key=None, configs=None,
                    cv_mode="vector", n_folds=5, n_procs=None, min_nprocs=4, timeout_sleep=3,
                    n_timeout=0, cpu_avail_thresh=30):
    """

    Parameters
    ----------
    regs
    Xtrain:
        The training input data in the format corresponding to `cv_mode`
    Ytrain:
        The output data, with `Ytrain` = (Ytrain_locs, Ytrain_obs), with Ytrain_locs and Yobs of len = n_samples
            and for 1 <= i <= n_samples, Ylocs[i] and Yobs[i] have shape = [n_observations_i, 1]
    Xtest:
        The testing input data in the format corresponding to `cv_mode`
    Ytest
    rec_path
    key
    configs: list or tuple
        The configurations dictionaries corresponding to
    cv_mode: {"discrete_func", "vector", "smooth_func"}
        The form of the input data
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

    """
    # Number of processors for execution
    if n_procs is None:
        n_procs = check_cpu_availability(min_nprocs=min_nprocs, timeout_sleep=timeout_sleep,
                                         n_timeout=n_timeout, cpu_avail_thresh=cpu_avail_thresh)
    # Instantiate cross validation
    cross_val = cross_validation.KfoldsCrossVal(n_folds=n_folds, mode=cv_mode)
    results = parallel_cross_vals(regs, Xtrain, Ytrain, cross_val, n_procs, rec_path=rec_path, key=key, configs=configs)
    best_ind = int(np.argmin(results))
    best_reg = regs[best_ind]
    best_reg.fit(Xtrain, Ytrain)
    preds = best_reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
    score_test = metrics.mse(preds, Ytest[1])
    if configs is not None:
        return configs[best_ind], results[best_ind], score_test
    else:
        return results[best_ind], score_test

