import numpy as np

from functional_data import smoothing
from functional_data import fpca
from functional_data.DEPRECATED import discrete_functional_data as disc_fd


class KernelAdditiveModel:

    def __init__(self, regu, kerlocs_in, kerlocs_out, kerevals, n_evals_in,
                 n_evals_out, n_fpca, domain_in, domain_out, n_evals_fpca):
        """
        Parameters
        ----------
        regu: float
            regularization parameter for the metho
        kerlocs_in: functional_regressors.kernels.ScalarKernel
            kernel for input locations comparison
        kerlocs_out: functional_regressors.kernels.ScalarKernel
            kernel for output locations comparison
        kerevals: functional_regressors.kernels.ScalarKernel
            kernel for input functions evaluation comparison
        n_evals_in: int
            number of evaluation points to use for scalar product approximation for input locations
        n_evals_out: int
            number of evaluation points to use for scalar product approximation for output locations
        n_fpca: int
            number of principal functions to use in approximation
        domain_in: array-like, shape=[1, 2]
            input domain
        domain_out: array-like, shape=[1, 2]
            output_domain
        n_evals_fpca: int
            number of evaluations to use in the discretized estimation of the FPCA
        """
        self.regu = regu
        self.kerlocs_in = kerlocs_in
        self.kerlocs_out = kerlocs_out
        self.kerevals = kerevals
        self.space_in = np.linspace(domain_in[0, 0], domain_in[0, 1], n_evals_in)
        self.space_out = np.linspace(domain_out[0, 0], domain_out[0, 1], n_evals_out)
        self.n_fpca = n_fpca
        self.domain_in = domain_in
        self.domain_out = domain_out
        self.fpca = fpca.FunctionalPCA(domain_out, n_evals_fpca, output_smoother=smoothing.LinearInterpSmoother())
        self.alpha = None
        self.Xfunc = None
        self.Ymean = None
        self.Yfpca = None

    @staticmethod
    def compute_Aevals(Xfunc0, Xfunc1, kerlocs_in, kerevals, approx_space, domain):
        Klocs_in = kerlocs_in(np.expand_dims(approx_space, axis=1), np.expand_dims(approx_space, axis=1))
        Xevals0 = np.array([f(approx_space) for f in Xfunc0])
        Xevals1 = np.array([f(approx_space) for f in Xfunc1])
        n = len(Xfunc0)
        m = len(Xfunc1)
        Aevals = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Kevals = kerevals(np.expand_dims(Xevals1[i], axis=1), np.expand_dims(Xevals0[j], axis=1))
                inter_const = domain[0, 1] - domain[0, 0]
                Aevals[i, j] = inter_const ** 2 * np.mean(Kevals * Klocs_in)
                # Aevals[i, j] = np.sum(Kevals * Klocs_in)
        return Aevals

    @staticmethod
    def compute_Aout(Yfpca, kerlocs_out, approx_space, domain):
        Klocs_out = kerlocs_out(np.expand_dims(approx_space, axis=1), np.expand_dims(approx_space, axis=1))
        n_fpca = len(Yfpca)
        Yfpca_evals = np.array([f(approx_space) for f in Yfpca])
        Afpca = np.zeros((n_fpca, n_fpca))
        inter_const = domain[0, 1] - domain[0, 0]
        for i in range(n_fpca):
            for j in range(n_fpca):
                Afpca[i, j] = inter_const ** 2 * (1 / approx_space.shape[0] ** 2) * Yfpca_evals[i].dot(Klocs_out).dot(Yfpca_evals[j])
        return Afpca

    @staticmethod
    def compute_Aout_pred(Yfpca, kerlocs_out, approx_space, pred_locs, domain):
        Klocs_out = kerlocs_out(np.expand_dims(pred_locs, axis=1), np.expand_dims(approx_space, axis=1))
        Aout_pred = np.zeros((pred_locs.shape[0], len(Yfpca)))
        Yfpca_evals = np.array([f(approx_space) for f in Yfpca])
        inter_const = domain[0, 1] - domain[0, 0]
        for i in range(len(Yfpca)):
            Aout_pred[:, i] = inter_const * (1 / approx_space.shape[0]) * Yfpca_evals[i].dot(Klocs_out)
        return Aout_pred

    @staticmethod
    def compute_Y(Yfunc, Yfpca, approx_space, domain):
        Yfpca_evals = np.array([f(approx_space) for f in Yfpca])
        Y_evals = np.array([f(approx_space) for f in Yfunc])
        inter_const = domain[0, 1] - domain[0, 0]
        return inter_const * (1 / approx_space.shape[0]) * Y_evals.dot(Yfpca_evals.T)

    def fit(self, X, Y, input_data_format="discrete_general", output_data_format="discrete_general"):
        Ywrapped = disc_fd.wrap_functional_data(Y, output_data_format)
        Xwrapped = disc_fd.wrap_functional_data(X, input_data_format)
        Xfunc = Xwrapped.func_linearinterp()
        self.Ymean = Ywrapped.mean_func()
        Ycentered = Ywrapped.centered_func_linearinterp()
        self.fpca.fit(Ycentered)
        Yfpca = self.fpca.get_regressors(self.n_fpca)
        A_evals_in = self.compute_Aevals(Xfunc, Xfunc, self.kerlocs_in, self.kerevals, self.space_in, self.domain_in)
        A_evals_fpca = self.compute_Aout(Yfpca, self.kerlocs_out, self.space_out, self.domain_out)
        self.Yfpca = Yfpca
        Y = self.compute_Y(Ycentered, Yfpca, self.space_out, self.domain_out)
        A = np.kron(A_evals_in, A_evals_fpca)
        self.Xfunc = Xfunc
        self.alpha = np.linalg.inv(A.T.dot(A) + self.regu * A).dot(A).dot(Y.flatten())
        self.alpha = self.alpha.reshape((len(Xfunc), len(Yfpca)))

    def predict_evaluate(self, Xnew, locs, input_data_format="discrete_general"):
        Xnew_wrapped = disc_fd.wrap_functional_data(Xnew, input_data_format)
        Xfunc_new = Xnew_wrapped.func_linearinterp()
        A_evals_in = self.compute_Aevals(self.Xfunc, Xfunc_new, self.kerlocs_in,
                                         self.kerevals, self.space_in, self.domain_in)
        A_evals_out = self.compute_Aout_pred(self.Yfpca, self.kerlocs_out,
                                             self.space_out, locs, self.domain_out)
        inter_prod = self.alpha.dot(A_evals_out.T)
        inter_prod2 = inter_prod.T.dot(A_evals_in.T)
        Ymean_eval = self.Ymean(locs)
        n = len(Xfunc_new)
        Ymean_evals = np.tile(Ymean_eval, (n, 1))
        return inter_prod2.T + Ymean_evals

    def predict_evaluate_diff_locs(self, Xnew, Yins_new, input_data_format="discrete_general"):
        n_new = len(Yins_new)
        return [self.predict_evaluate(([Xnew[0][i]], [Xnew[1][i]]), Yins_new[i], input_data_format) for i in range(n_new)]
