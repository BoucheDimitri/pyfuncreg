from abc import ABC, abstractmethod


class FunctionalRegressor(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, Y):
        """
        Fit the functional regressor

        Parameters
        ----------
        X:
            Input data, form can vary
        Y:
            Functional output data, form can vary
        """
        pass

    @abstractmethod
    def predict(self, Xnew):
        """
        Predict with function as output

        Parameters
        ----------
        Xnew:
            New input data, form can vary
        """
        pass

    @abstractmethod
    def predict_evaluate(self, Xnew, locs):
        """
        Predict the function and evaluate it at `locs`

        Parameters
        ----------
        Xnew:
            New input data, form can vary
        locs:
            The locations of evaluation
        """
        pass

    @abstractmethod
    def predict_evaluate_diff_locs(self, Xnew, Ylocs_new):
        """
        Predict the function and evaluate it at a different loc for each input

        Parameters
        ----------
        Xnew:
            New input data, form can vary
        Ylocs_new:
            The list of locations of evaluations
        """
        pass
