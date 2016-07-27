import time
from scipy.optimize import minimize
from fitted_model import Fitted_Model

class Nelder_Mead_Algo:
    def __init__(self, data, settings=None):
        self.data = data
        self.settings = settings
        self._create_problem_wrapper()

    def run(self, initial_lambdas_set, num_iters=10, log_file=None):
        self.log_file = log_file
        start = time.time()
        total_calls = 0
        self.fmodel = Fitted_Model(initial_lambdas_set[0].size)
        for initial_lambdas in initial_lambdas_set:
            res = minimize(self.get_validation_cost, initial_lambdas, method='nelder-mead', options={"maxiter":num_iters})
            model_params = self.problem_wrapper.solve(res.x, quick_run=True)
            total_calls += res.nfev
            self.fmodel.update(res.x, model_params, res.fun)
            self.log("fmodel %s" % self.fmodel)
        runtime = time.time() - start

        self.log("%s: best cost %f, lambda %s, total calls %d" % (self.method_label, self.fmodel.best_cost, self.fmodel.best_lambdas, total_calls))

        self.fmodel.set_runtime(runtime)

    def log(self, log_str):
        if self.log_file is None:
            print log_str
        else:
            self.log_file.write("%s\n" % log_str)
