import time
import itertools
import numpy as np
import scipy as sp
from common import *
from convexopt_solvers import Lambda12ProblemWrapper
from scipy.optimize import minimize

def run(X_train, y_train, X_validate, y_validate, diminishing_step_size=False, initial_lambda1=1, initial_lambda2=1):
    start = time.time()

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)

    def get_validation_cost(lambda_pair):
        if lambda_pair[0] <= 0 or lambda_pair[1] <= 0:
            return 10000

        beta_guess = problem_wrapper.solve(lambda_pair[0], lambda_pair[1])
        current_cost = testerror(X_validate, y_validate, beta_guess)
        return current_cost

    res = minimize(get_validation_cost, (1,1), method='nelder-mead')

    runtime = time.time() - start

    best_beta = problem_wrapper.solve(res.x[0], res.x[1])

    return best_beta, runtime
