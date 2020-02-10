from scipy import optimize


class ScipySolver:

    def __init__(self, tol, maxit, method="BFGS"):
        self.tol = tol
        self.maxit = maxit
        self.obj_record = []
        self.method = method

    def __call__(self, fun, x0, jac):
        obj_records = []

        def callback(xk):
            obj_records.append(fun(xk))

        sol = optimize.minimize(fun, x0, method=self.method, jac=jac,
                                tol=self.tol, options={"maxiter": self.maxit}, callback=callback)

        return sol, obj_records