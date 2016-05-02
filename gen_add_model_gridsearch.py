import sys
import numpy as np

from common import *
from convexopt_solvers import GenAddModelProblemWrapper

LAMBDA_MIN_FACTOR = 1e-8
MAX_LAMBDA = 2

def run(X_train, y_train, X_validate, y_validate, num_lambdas=10):
    max_power = np.log(MAX_LAMBDA)
    min_power = np.log(LAMBDA_MIN_FACTOR * max_power)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 0.01) / (num_lambdas - 1)))
    print "gridsearch: lambda_guesses", lambda_guesses

    num_features = X_train.shape[1]
    full_X = np.vstack((X_train, X_validate))
    num_samples = full_X.shape[0]
    train_indices = np.arange(X_train.shape[0])
    test_indices = np.arange(X_train.shape[0], num_samples)
    problem_wrapper = GenAddModelProblemWrapper(full_X, train_indices, y_train)

    best_cost = 1e5
    best_thetas = []
    best_regularization = [lambda_guesses[0]] * num_features

    for l in lambda_guesses:
        regularization = [l] * num_features
        thetas = problem_wrapper.solve(regularization)
        current_cost = testerror_multi_smooth(y_validate, test_indices, thetas)
        if best_cost > current_cost:
            best_cost = current_cost
            best_thetas = thetas
            best_regularization = regularization
            print "best_cost so far", best_cost, "best_regularization", best_regularization

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_regularization

    x_order = np.argsort(full_X[:,0])
    y_full = np.vstack((y_train, y_validate))
    print np.hstack((full_X[x_order,:], best_thetas, y_full[x_order, :]))

    return best_thetas, best_cost
