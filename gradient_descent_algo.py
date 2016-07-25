import sys
import time
import scipy as sp
from fitted_model import Fitted_Model
import numpy as np
import cvxpy

from common import *

class Gradient_Descent_Algo:
    def __init__(self, data):
        self.data = data

        self._create_descent_settings()
        self._create_problem_wrapper()
        self._create_lambda_configs()

    def run(self, initial_lambda_set, debug=True):
        start_time = time.time()

        self.fmodel = Fitted_Model(initial_lambda_set[0].size)
        best_cost = None
        for initial_lambdas in initial_lambda_set:
            self._run_lambdas(initial_lambdas, debug=debug)
            if best_cost != self.fmodel.best_cost:
                print "%s: best start lambda %s" % (self.method_label, initial_lambdas)

        runtime = time.time() - start_time
        print "%s: runtime %s" % (self.method_label, runtime)
        self.fmodel.set_runtime(runtime)

    def _run_lambdas(self, initial_lambdas, debug=True):
        print "%s: initial_lambdas %s" % (self.method_label, initial_lambdas)
        start_history_idx = len(self.fmodel.cost_history)
        # warm up the problem
        self.problem_wrapper.solve(initial_lambdas, quick_run=True)
        # do a real run now
        model_params = self.problem_wrapper.solve(initial_lambdas, quick_run=False)
        # Check that no model params are None
        assert(not self._any_model_params_none(model_params))

        current_cost = self.get_validate_cost(model_params)
        self.fmodel.update(initial_lambdas, model_params, current_cost)
        print "self.fmodel.current_cost", self.fmodel.current_cost

        step_size = self.step_size_init
        for i in range(0, self.num_iters):
            lambda_derivatives = self._get_lambda_derivatives()

            if debug:
                self._double_check_derivative(lambda_derivatives)

            potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                step_size,
                lambda_derivatives,
                quick_run=True
            )

            # TODO: Do backtracking
            while self._check_should_backtrack(potential_cost, step_size, lambda_derivatives) and step_size > self.step_size_min:
                if potential_cost is None: # Then cvxpy couldn't find a solution. Shrink faster
                    step_size *= self.shrink_factor**3
                else:
                    step_size *= self.shrink_factor
                potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                    step_size,
                    lambda_derivatives,
                    quick_run=True
                )
                if potential_cost is not None:
                    print "(shrinking) potential_lambdas %s, cost %f, step, %f" % (potential_lambdas, potential_cost, step_size)
                else:
                    print "(shrinking) potential_lambdas None!"

            if self.fmodel.current_cost < potential_cost:
                print "COST IS INCREASING!", potential_cost
                break
            else:
                potential_lambdas, potential_model_params, potential_cost = self._run_potential_lambdas(
                    step_size,
                    lambda_derivatives,
                    quick_run=False
                )
                self.fmodel.update(potential_lambdas, potential_model_params, potential_cost)

                print self.method_label, "iter:", i, "step_size", step_size
                print "current model", self.fmodel
                print "cost_history", self.fmodel.cost_history[start_history_idx:]

                if self.fmodel.get_cost_diff() < self.decr_enough_threshold:
                    print "decrease amount too small", self.fmodel.get_cost_diff()
                    break

            if step_size < self.step_size_min:
                print self.method_label, "STEP SIZE TOO SMALL", step_size
                break
            sys.stdout.flush()

        print "TOTAL ITERS", i
        print self.fmodel.cost_history[start_history_idx:]

    def _check_should_backtrack(self, potential_cost, step_size, lambda_derivatives):
        if potential_cost is None:
            return True
        backtrack_thres_raw = self.fmodel.current_cost - self.backtrack_alpha * step_size * np.linalg.norm(lambda_derivatives)**2
        backtrack_thres = self.fmodel.current_cost if backtrack_thres_raw < 0 else backtrack_thres_raw
        return potential_cost > backtrack_thres

    def _run_potential_lambdas(self, step_size, lambda_derivatives, quick_run=False):
        potential_lambdas = self._get_updated_lambdas(
            step_size,
            lambda_derivatives
        )
        try:
            potential_model_params = self.problem_wrapper.solve(potential_lambdas, quick_run=quick_run)
        except cvxpy.error.SolverError:
            potential_model_params = None

        if self._any_model_params_none(potential_model_params):
            potential_cost = None
        else:
            potential_cost = self.get_validate_cost(potential_model_params)
        return potential_lambdas, potential_model_params, potential_cost

    def _get_updated_lambdas(self, method_step_size, lambda_derivatives):
        current_lambdas = self.fmodel.current_lambdas
        new_step_size = method_step_size
        if self.use_boundary:
            potential_lambdas = current_lambdas - method_step_size * lambda_derivatives

            for idx in range(0, current_lambdas.size):
                if current_lambdas[idx] > self.lambda_mins[idx] and potential_lambdas[idx] < (1 - self.boundary_factor) * current_lambdas[idx]:
                    smaller_step_size = self.boundary_factor * current_lambdas[idx] / lambda_derivatives[idx]
                    new_step_size = min(new_step_size, smaller_step_size)
                    print self.method_label, "USING THE BOUNDARY!", new_step_size

        return np.maximum(current_lambdas - new_step_size * lambda_derivatives, self.lambda_mins)

    @staticmethod
    def _any_model_params_none(model_params):
        if model_params is None:
            return True
        else:
            return any([m is None for m in model_params])
