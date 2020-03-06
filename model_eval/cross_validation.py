import numpy as np
from sklearn.model_selection import KFold

from functional_data.DEPRECATED import discrete_functional_data

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
                 input_format_fit="vector", output_format_fit="discrete_samelocs_regular_1d",
                 input_format_test="vector"):
        self.score_func = score_func
        self.n_folds = n_folds
        self.seed = seed
        self.cross_val = KFold(random_state=seed, shuffle=shuffle, n_splits=n_folds)
        self.input_format_fit = input_format_fit
        self.output_format_fit = output_format_fit
        self.input_format_test = input_format_test

    @staticmethod
    def get_subset(data, index, data_format="vector"):
        if data_format == "vector":
            return data[index]
        elif data_format == "discrete_general":
            return [data[0][i] for i in index], [data[1][i] for i in index]
        elif data_format == "discrete_samelocs_regular_1d":
            return data[0], data[1][index]
        else:
            raise ValueError("Must chose mode in {'vector', 'discrete_general', 'discrete_samelocs_regular_1d'}")

    # TODO: FINIR ADAPTATION AVEC INPUT FORMAT TEST
    # TODO: NON EN FAIT PAS FORCEMENT NECESSAIRE PUISQUE C EST QUE DU TRAIN ICI
    def __call__(self, reg, X, Y):
        scores = []
        if self.input_format_fit == "vector":
            n_samples = len(X)
        else:
            n_samples = len(X[1])
        inds_split = self.cross_val.split(np.zeros((n_samples, 1)))
        for train_index, test_index in inds_split:
            # Select subsets using the indexing adapted for the data format
            Xsub_train = KfoldsCrossVal.get_subset(X, train_index, data_format=self.input_format_fit)
            Ysub_train = KfoldsCrossVal.get_subset(Y, train_index, data_format=self.output_format_fit)
            Xsub_test = KfoldsCrossVal.get_subset(X, test_index, data_format=self.input_format_fit)
            # Put testing output data in discrete general form (same format for all regressors)
            Ysub_test = KfoldsCrossVal.get_subset(Y, test_index, data_format=self.output_format_fit)
            Ysub_test = discrete_functional_data.to_discrete_general(Ysub_test, self.output_format_fit)
            # Fit on the training subset
            reg.fit(Xsub_train, Ysub_train, self.input_format_fit, self.output_format_fit)
            # Predict on validation subset
            preds = reg.predict_evaluate_diff_locs(Xsub_test, Ysub_test[0], input_data_format=self.input_format_fit)
            scores.append(self.score_func(preds, Ysub_test[1]))
        return np.mean(scores)