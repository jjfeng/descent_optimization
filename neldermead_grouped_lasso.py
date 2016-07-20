
import time
import itertools
import numpy as np
import scipy as sp
from common import *
from convexopt_solvers import GroupedLassoProblemWrapper
from convexopt_solvers import GroupedLassoProblemWrapperSimple
# from convexopt_solvers import GroupedLassoClassifyProblemWrapperSimple
# from convexopt_solvers import GroupedLassoClassifyProblemWrapperSimpleFullCV
from scipy.optimize import minimize

def run(X_train, y_train, X_validate, y_validate, group_feature_sizes, pooled=True):
    start = time.time()

    if pooled:
        problem_wrapper = GroupedLassoProblemWrapperSimple(X_train, y_train, group_feature_sizes)
        init_regularizations = [1,1]
    else:
        # Note: this is really slow
        init_regularizations = np.ones(len(group_feature_sizes) + 1)
        problem_wrapper = GroupedLassoProblemWrapper(X_train, y_train, group_feature_sizes)

    def get_validation_cost(regularization):
        if np.any(regularization < 0):
            return 10000

        betas = problem_wrapper.solve(regularization)

        validation_cost = testerror_grouped(X_validate, y_validate, betas)
        return validation_cost

    res = minimize(get_validation_cost, (init_regularizations), method='nelder-mead')

    runtime = time.time() - start

    best_beta = problem_wrapper.solve(res.x)

    return best_beta, runtime
