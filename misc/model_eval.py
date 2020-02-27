import numpy as np
from sklearn.model_selection import KFold
import multiprocessing
import time
import pickle
import psutil

from model_eval import cross_validation


def mean_squared_error(ytrue, ypred):
    scores = np.array([np.mean(np.linalg.norm(np.squeeze(ytrue[i]) - np.squeeze(ypred[i]))**2) for i in range(len(ytrue))])
    return np.mean(scores)


class CrossValidationEval:
    """
    Cross validation for sparsely observed functional data
    """
    def __init__(self, score_func=mean_squared_error, n_folds=5, seed=0, shuffle=False, pred_diff_locs=False):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)
        self.pred_diff_locs = pred_diff_locs

    def __call__(self, clf, X, Y):
        scores = []
        count = 0
        for train_index, test_index in self.cross_val.split(X):
            if self.pred_diff_locs:
                if isinstance(X, np.ndarray):
                    clf.fit(X[train_index], ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                    preds = clf.predict_evaluate_diff_locs(X[test_index], Y[0])
                    scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
                else:
                    clf.fit([X[i] for i in train_index],
                            ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                    preds = clf.predict_evaluate_diff_locs([X[i] for i in test_index], [Y[0][i] for i in test_index])
                    scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
            else:
                if isinstance(X, np.ndarray):
                    clf.fit(X[train_index], ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                    preds = [clf.predict_evaluate(np.expand_dims(X[i], axis=0), Y[0][i]) for i in test_index]
                    scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
                else:
                    clf.fit([X[i] for i in train_index], ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                    preds = [clf.predict_evaluate([X[i]], Y[0][i]) for i in test_index]
                    scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
            count += 1
        return np.mean(scores)


class CrossValidationEvalLocs:
    """
    Cross validation for sparsely observed functional data with locations varying
    """
    def __init__(self, score_func=mean_squared_error, n_folds=5, seed=0, shuffle=False):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)

    def __call__(self, clf, X, Y):
        scores = []
        count = 0
        inds_split = self.cross_val.split(np.array(X[0]).squeeze())
        for train_index, test_index in inds_split:
            clf.fit(([X[0][i] for i in train_index], [X[1][i] for i in train_index]),
                    ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
            preds = [clf.predict_evaluate(([X[0][i]], [X[1][i]]), Y[0][i]) for i in test_index]
            scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
            count += 1
        return np.mean(scores)


def exec_regressors_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
                          rec_path=None, key="", nprocs=None, timeout=0, minnprocs=4, eval_diff_locs=False):
    """
    Parallelized cross-validation of regressors
    """
    if nprocs is None:
        cpu_occup = np.array(psutil.cpu_percent(percpu=True))
        nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
        timeout_count = 0
        while nprocs < minnprocs and timeout_count < timeout:
            time.sleep(3)
            cpu_occup = np.array(psutil.cpu_percent(percpu=True))
            nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
            timeout_count += 1
    nexpes = len(regressors)
    nsplits = len(regressors) // nprocs
    results = []
    cross_val = CrossValidationEval(pred_diff_locs=eval_diff_locs)
    if nsplits == 0:
        with multiprocessing.Pool(processes=nexpes) as pool:
            multiple_results = [pool.apply_async(cross_val, (regressors[i], Xtrain, Ytrain, ))
                                for i in range(nexpes)]
            results += [res.get() for res in multiple_results]
    else:
        count = 0
        for i in range(nsplits + 1):
            if count < nsplits:
                regressors_split = regressors[i * nprocs: (i + 1) * nprocs]
                with multiprocessing.Pool(processes=nprocs) as pool:
                    multiple_results = [pool.apply_async(cross_val,
                                                         (regressors_split[j], Xtrain, Ytrain, ))
                                        for j in range(len(regressors_split))]
                    results += [res.get() for res in multiple_results]
                print(
                    "Process batch number " + str(count) + " finished. Remaining: " + str(nsplits - count - 1))
                if rec_path is not None:
                    with open(rec_path + "/batch_no" + str(count) + "out_of" + str(nsplits) + "_" + key + ".pkl", "wb") as outp:
                        pickle.dump((expe_dicts[:(i + 1) * nprocs], results[:(i + 1) * nprocs]), outp,
                                    pickle.HIGHEST_PROTOCOL)
                count += 1
            else:
                regressors_split = regressors[i * nprocs:]
                if len(regressors_split) > 0:
                    with multiprocessing.Pool(processes=len(regressors_split)) as pool:
                        multiple_results = [pool.apply_async(cross_val,
                                                            (regressors_split[j], Xtrain, Ytrain, ))
                                            for j in range(len(regressors_split))]
                        results += [res.get() for res in multiple_results]
    best_ind = np.argmin(results)
    best_regressor = regressors[best_ind]
    best_regressor.fit(Xtrain, Ytrain)
    if isinstance(Xtrain, np.ndarray):
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
    elif len(Xtest) == 2:
        len_test = len(Xtest[0])
        preds = [best_regressor.predict_evaluate([Xtest[i]], Ytest[0][i])
                 for i in range(len_test)]
    else:
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate([Xtest[i]], Ytest[0][i])
                 for i in range(len_test)]
    score_test = mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
    return expe_dicts, results, best_ind, expe_dicts[best_ind], results[best_ind], score_test


def exec_regressors_queue_bis(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest,
                              rec_path=None, key="", nprocs=None, timeout=0, minnprocs=4, cv_mode="vector"):
    """
    Parallelized cross-validation of regressors
    """
    if nprocs is None:
        cpu_occup = np.array(psutil.cpu_percent(percpu=True))
        nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
        timeout_count = 0
        while nprocs < minnprocs and timeout_count < timeout:
            time.sleep(3)
            cpu_occup = np.array(psutil.cpu_percent(percpu=True))
            nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
            timeout_count += 1
    nexpes = len(regressors)
    nsplits = len(regressors) // nprocs
    results = []
    cross_val = cross_validation.KfoldsCrossVal(mode=cv_mode)
    if nsplits == 0:
        with multiprocessing.Pool(processes=nexpes) as pool:
            multiple_results = [pool.apply_async(cross_val, (regressors[i], Xtrain, Ytrain, ))
                                for i in range(nexpes)]
            results += [res.get() for res in multiple_results]
    else:
        count = 0
        for i in range(nsplits + 1):
            if count < nsplits:
                regressors_split = regressors[i * nprocs: (i + 1) * nprocs]
                with multiprocessing.Pool(processes=nprocs) as pool:
                    multiple_results = [pool.apply_async(cross_val,
                                                         (regressors_split[j], Xtrain, Ytrain, ))
                                        for j in range(len(regressors_split))]
                    results += [res.get() for res in multiple_results]
                print(
                    "Process batch number " + str(count) + " finished. Remaining: " + str(nsplits - count - 1))
                if rec_path is not None:
                    with open(rec_path + "/batch_no" + str(count) + "out_of" + str(nsplits) + "_" + key + ".pkl", "wb") as outp:
                        pickle.dump((expe_dicts[:(i + 1) * nprocs], results[:(i + 1) * nprocs]), outp,
                                    pickle.HIGHEST_PROTOCOL)
                count += 1
            else:
                regressors_split = regressors[i * nprocs:]
                if len(regressors_split) > 0:
                    with multiprocessing.Pool(processes=len(regressors_split)) as pool:
                        multiple_results = [pool.apply_async(cross_val,
                                                            (regressors_split[j], Xtrain, Ytrain, ))
                                            for j in range(len(regressors_split))]
                        results += [res.get() for res in multiple_results]
    best_ind = np.argmin(results)
    best_regressor = regressors[best_ind]
    best_regressor.fit(Xtrain, Ytrain)
    if isinstance(Xtrain, np.ndarray):
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate(np.expand_dims(Xtest[i], axis=0), Ytest[0][i])
                 for i in range(len_test)]
    elif len(Xtest) == 2:
        len_test = len(Xtest[0])
        preds = [best_regressor.predict_evaluate([Xtest[i]], Ytest[0][i])
                 for i in range(len_test)]
    else:
        len_test = len(Xtest)
        preds = [best_regressor.predict_evaluate([Xtest[i]], Ytest[0][i])
                 for i in range(len_test)]
    score_test = mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
    return expe_dicts, results, best_ind, expe_dicts[best_ind], results[best_ind], score_test


def exec_regressors_eval_queue(regressors, expe_dicts, Xtrain, Ytrain, Xtest, Ytest, nprocs=None,
                               rec_path=None, key="", timeout=0, minnprocs=4):
    """
    Parallelized cross-validation of regressors
    """
    if nprocs is None:
        cpu_occup = np.array(psutil.cpu_percent(percpu=True))
        nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
        timeout_count = 0
        while nprocs < minnprocs and timeout_count < timeout:
            time.sleep(3)
            cpu_occup = np.array(psutil.cpu_percent(percpu=True))
            nprocs = (cpu_occup[cpu_occup < 90]).shape[0]
            timeout_count += 1
    nexpes = len(regressors)
    nsplits = len(regressors) // nprocs
    results = []
    cross_val = CrossValidationEvalLocs()
    if nsplits == 0:
        with multiprocessing.Pool(processes=nexpes) as pool:
            multiple_results = [pool.apply_async(cross_val, (regressors[i], Xtrain, Ytrain, ))
                                for i in range(nexpes)]
            results += [res.get() for res in multiple_results]
    else:
        count = 0
        for i in range(nsplits + 1):
            if count < nsplits:
                regressors_split = regressors[i * nprocs: (i + 1) * nprocs]
                with multiprocessing.Pool(processes=nprocs) as pool:
                    multiple_results = [pool.apply_async(cross_val,
                                                         (regressors_split[j], Xtrain, Ytrain, ))
                                        for j in range(len(regressors_split))]
                    results += [res.get() for res in multiple_results]
                print(
                    "Process batch number " + str(count) + " finished. Remaining: " + str(nsplits - count - 1))
                if rec_path is not None:
                    with open(rec_path + "/batch_no" + str(count) + "out_of" + str(nsplits) + "_" + key + ".pkl", "wb") as outp:
                        pickle.dump((expe_dicts[:(i + 1) * nprocs], results[:(i + 1) * nprocs]), outp,
                                    pickle.HIGHEST_PROTOCOL)
                count += 1
            else:
                regressors_split = regressors[i * nprocs:]
                if len(regressors_split) > 0:
                    with multiprocessing.Pool(processes=len(regressors_split)) as pool:
                        multiple_results = [pool.apply_async(cross_val,
                                                         (regressors_split[j], Xtrain, Ytrain, ))
                                            for j in range(len(regressors_split))]
                        results += [res.get() for res in multiple_results]
    best_ind = np.argmin(results)
    best_regressor = regressors[best_ind]
    best_regressor.fit(Xtrain, Ytrain)
    len_test = len(Xtest[0])
    preds = [best_regressor.predict_evaluate(([Xtest[0][i]], [Xtest[1][i]]), Ytest[0][i]) for i in range(len_test)]
    score_test = mean_squared_error(preds, [Ytest[1][i] for i in range(len_test)])
    return expe_dicts, results, best_ind, expe_dicts[best_ind], results[best_ind], score_test
