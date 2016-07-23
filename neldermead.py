import time
from scipy.optimize import minimize
from fitted_model import Fitted_Model

class Nelder_Mead_Algo:
    def __init__(self, data):
        self.data = data
        self._create_problem_wrapper()

    def run(self, initial_lambdas, num_iters=10):
        start = time.time()

        res = minimize(self.get_validation_cost, initial_lambdas, method='nelder-mead', options={"maxiter":num_iters})

        runtime = time.time() - start

        print "%s: best cost %f, lambda %s, total calls %d" % (self.method_label, res.fun, res.x, res.nfev)
        best_model_params = self.problem_wrapper.solve(res.x)

        self.fmodel = Fitted_Model(initial_lambdas.size)
        self.fmodel.update(res.x, best_model_params, res.fun)
        self.fmodel.set_runtime(runtime)
