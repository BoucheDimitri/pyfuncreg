import numpy as np
from sklearn.model_selection import KFold

from functional_data import discrete_functional_data as disc_fd

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
    def __init__(self, score_func=metrics.mse, n_folds=5, shuffle=False, seed=0,
                 input_indexing="discrete_general", output_indexing="discrete_general"):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)
        self.input_indexing = input_indexing
        self.output_indexing = output_indexing

    @staticmethod
    def get_subset(data, index, input_indexing="array"):
        if input_indexing == "array":
            return data[index]
        elif input_indexing == "discrete_general":
            return [data[0][i] for i in index], [data[1][i] for i in index]
        elif input_indexing == 'list':
            return [data[i] for i in index]
        else:
            raise ValueError("Must chose mode in {'array', 'list', 'discrete_general'}")

    def __call__(self, reg, Xfit, Yfit, Xpred=None, Ypred=None):
        scores = []
        if self.input_indexing == "array":
            n_samples = len(Xfit)
        else:
            n_samples = len(Xfit[1])
        inds_split = self.cross_val.split(np.zeros((n_samples, 1)))
        if Xpred is None:
            Xpred = Xfit
        if Ypred is None:
            Ypred = Yfit
        for train_index, test_index in inds_split:
            # Select subsets using the indexing adapted for the data format
            Xtrain = KfoldsCrossVal.get_subset(Xfit, train_index, self.input_indexing)
            Ytrain = KfoldsCrossVal.get_subset(Yfit, train_index, self.output_indexing)
            Xtest = KfoldsCrossVal.get_subset(Xpred, test_index, self.input_indexing)
            # Put testing output data in discrete general form (same format for all regressors)
            Ytest = KfoldsCrossVal.get_subset(Ypred, test_index, self.output_indexing)
            Ytest = disc_fd.to_discrete_general(*Ytest)
            # Fit on the training subset
            reg.fit(Xtrain, Ytrain)
            # Predict on validation subset
            preds = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
            scores.append(self.score_func(preds, Ytest[1]))
        return np.mean(scores)


class KfoldsCrossValRegpath:
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
    def __init__(self, score_func=metrics.mse, n_folds=5, shuffle=False, seed=0,
                 input_indexing="discrete_general", output_indexing="discrete_general"):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)
        self.input_indexing = input_indexing
        self.output_indexing = output_indexing

    @staticmethod
    def get_subset(data, index, input_indexing="array"):
        if input_indexing == "array":
            return data[index]
        elif input_indexing == "discrete_general":
            return [data[0][i] for i in index], [data[1][i] for i in index]
        elif input_indexing == 'list':
            return [data[i] for i in index]
        else:
            raise ValueError("Must chose mode in {'array', 'list', 'discrete_general'}")

    def __call__(self, reg, Xfit, Yfit, Xpred=None, Ypred=None):
        scores = []
        if self.input_indexing == "array":
            n_samples = len(Xfit)
        else:
            n_samples = len(Xfit[1])
        inds_split = self.cross_val.split(np.zeros((n_samples, 1)))
        if Xpred is None:
            Xpred = Xfit
        if Ypred is None:
            Ypred = Yfit
        for train_index, test_index in inds_split:
            # Select subsets using the indexing adapted for the data format
            Xtrain = KfoldsCrossVal.get_subset(Xfit, train_index, self.input_indexing)
            Ytrain = KfoldsCrossVal.get_subset(Yfit, train_index, self.output_indexing)
            Xtest = KfoldsCrossVal.get_subset(Xpred, test_index, self.input_indexing)
            # Put testing output data in discrete general form (same format for all regressors)
            Ytest = KfoldsCrossVal.get_subset(Ypred, test_index, self.output_indexing)
            Ytest = disc_fd.to_discrete_general(*Ytest)
            # Fit on the training subset
            reg.fit(Xtrain, Ytrain)
            # Predict on validation subset
            preds = reg.predict_evaluate_diff_locs(Xtest, Ytest[0])
            scores.append(self.score_func(preds, Ytest[1]))
        return np.mean(scores)