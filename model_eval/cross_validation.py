import numpy as np
from sklearn.model_selection import KFold


from model_eval import metrics


class KfoldsCrossVal:
    """
    Kfolds cross validation for functional data

    Parameters
    ----------
    score_func: callable
        Score metric
    n_folds: int
        Number of folds
    shuffle: bool
        Should the indices be shuffled prior to cross-val
    seed: int
        Seed for the shuffling if it is set to True
    mode: {"discrete_func", "vector", "smooth_func"}
        The form of the input data
    """
    def __init__(self, score_func=metrics.mse, n_folds=5, shuffle=False,
                 seed=0, mode="discrete_func"):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)
        self.mode = mode

    def discrete_func_call(self, reg, X, Y):
        """
        Cross validation score when mode is set to "discrete_func"

        Parameters
        ----------
        reg: functional_regressors.functional_regressor.FunctionalRegressor
            The regressors
        X: tuple or list, len = 2
            The input data, with X = (Xlocs, Xobs), with Xlocs and Xobs of len = n_samples
            and for 1 <= i <= n_samples, Xlocs[i] and Xobs[i] have shape = [n_observations_i, 1]
        Y: tuple or list, len = 2
            The output data, with Y = (Ylocs, Yobs), with Ylocs and Yobs of len = n_samples
            and for 1 <= i <= n_samples, Ylocs[i] and Yobs[i] have shape = [n_observations_i, 1]

        Returns
        -------
        float
            Cross-validation score
        """
        scores = []
        n = len(X[0])
        inds_split = self.cross_val.split(np.zeros((n, 1)))
        for train_index, test_index in inds_split:
            reg.fit(([X[0][i] for i in train_index], [X[1][i] for i in train_index]),
                    ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
            preds = reg.predict_evaluate_diff_locs(([X[0][i] for i in test_index], [X[1][i] for i in test_index]),
                                                    [Y[0][i] for i in test_index])
            scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
        return np.mean(scores)

    def vector_call(self, reg, X, Y):
        """
        Cross validation score when mode is set to "vector"

        Parameters
        ----------
        reg: functional_regressors.functional_regressor.FunctionalRegressor
            The regressors
        X: array-like, len = n_samples
            The input data
        Y: tuple or list, len = 2
            X = (Ylocs, Yobs), with Ylocs and Yobs of len = n_samples
            and for 1 <= i <= n_samples, Ylocs[i] and Yobs[i] have shape = [n_observations_i, 1]

        Returns
        -------
        float
            Cross-validation score
        """
        scores = []
        for train_index, test_index in self.cross_val.split(X):
            if isinstance(X, np.ndarray):
                reg.fit(X[train_index], ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                preds = reg.predict_evaluate_diff_locs(X[test_index], Y[0])
                scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
            else:
                reg.fit([X[i] for i in train_index],
                        ([Y[0][i] for i in train_index], [Y[1][i] for i in train_index]))
                preds = reg.predict_evaluate_diff_locs([X[i] for i in test_index], [Y[0][i] for i in test_index])
                scores.append(self.score_func(preds, [Y[1][i] for i in test_index]))
        return np.mean(scores)

    def __call__(self, reg, X, Y):
        """
        Wrapper for the calls using different modes

        Parameters
        ----------
        reg: functional_regressors.functional_regressor.FunctionalRegressor
            The regressor to cross-validate
        X:
            The input data in the form corresponding to the mode
        Y:
            The output data in the form corresponding to the mode

        Returns
        -------

        """
        if self.mode == "discrete_func":
            return self.discrete_func_call(reg, X, Y)
        elif self.mode == "vector":
            return self.vector_call(reg, X, Y)
        elif self.mode == "smooth_func":
            raise ValueError('The mode "smooth_func" has yet to be implemented')
        else:
            raise ValueError('Possible modes are {"discrete_func", "vector", "smooth_func"}')
