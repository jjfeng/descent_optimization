import sys
import numpy as np

from common import *
from convexopt_solvers import LassoClassifyProblemWrapper

from realdata_colitis_models import AllKFoldsData

LAMBDA_MIN_FACTOR = 1e-5
NUM_LAMBDAS = 20

def run_classify(X_groups_train, y_train, X_groups_validate, y_validate):
    """
    Although this function is given groups, it actually doesn't utilize the groups at all in the criterion
    """

    method_label = "gridsearch_lasso"

    X_validate = np.hstack(X_groups_validate)

    max_power = np.log(50)
    min_power = np.log(1e-4)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 1e-5) / (NUM_LAMBDAS - 1)))
    print method_label, "lambda_guesses", lambda_guesses

    X_train = np.hstack(X_groups_train)
    problem_wrapper = LassoClassifyProblemWrapper(X_train, y_train, [])

    best_cost = 1e5
    best_betas = []
    best_regularization = lambda_guesses[0]

    for l1 in reversed(lambda_guesses):
        betas = problem_wrapper.solve([l1])
        current_cost, _ = testerror_logistic_grouped(X_validate, y_validate, betas)
        if best_cost > current_cost:
            best_cost = current_cost
            best_betas = betas
            best_regularization = l1
            print method_label, "best_cost so far", best_cost, "best_regularization", best_regularization
            sys.stdout.flush()

    print method_label, "best_validation_error", best_cost
    print method_label, "best lambdas:", best_regularization

    return best_betas, best_cost


def run_classify_fullcv(X_groups_train_validate, y_train_validate, feature_group_sizes, kfolds):
    """
    Although this function is given groups, it actually doesn't utilize the groups at all in the criterion
    """

    method_label = "gridsearch_lasso_fullcv"

    num_lambdas = 6
    max_power = np.log(5)
    min_power = np.log(1e-4)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 1e-5) / (num_lambdas - 1)))
    print method_label, "lambda_guesses", lambda_guesses

    X_train_validate = np.hstack(X_groups_train_validate)
    full_problem = LassoClassifyProblemWrapper(X_train_validate, y_train_validate, feature_group_sizes)
    all_kfolds_data = AllKFoldsData(X_train_validate, y_train_validate, feature_group_sizes, kfolds, LassoClassifyProblemWrapper)

    best_cost = 1e5
    best_regularization = lambda_guesses[0]
    for l1 in reversed(lambda_guesses):
        betas, cost = all_kfolds_data.solve([l1])
        if best_cost > cost:
            best_cost = cost
            best_regularization = l1
            print method_label, "best_cost so far", best_cost, "best_regularization", best_regularization
            sys.stdout.flush()

    print method_label, "best_validation_error", best_cost
    print method_label, "best lambdas:", best_regularization

    betas = full_problem.solve([best_regularization])

    return betas, best_cost
