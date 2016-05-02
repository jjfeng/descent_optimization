import numpy as np

from common import *
from convexopt_solvers import EffectsInteractionProblemWrapperSimple

LAMBDA_EIGEN_FACTOR = 2
NUM_LAMBDAS = 15

def run(X_train, W_train, y_train, X_validate, W_validate, y_validate):
    min_power = np.log(1e-5)
    max_power = np.log(100)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power) / NUM_LAMBDAS))

    # Create the problem - allows for warmstart
    problem_wrapper = EffectsInteractionProblemWrapperSimple(X_train, W_train, y_train)

    best_beta = []   # initialize
    best_theta = []   # initialize
    best_cost = 1e5  # initialize to something huge
    best_lambdas = [lambda_guesses[0], lambda_guesses[0]]

    for l1 in lambda_guesses:
        for l2 in lambda_guesses:
            beta_guess, theta_guess = problem_wrapper.solve([l1, l2])
            current_cost = testerror_interactions(X_validate, W_validate, y_validate, beta_guess, theta_guess)

            if best_cost > current_cost:
                best_cost = current_cost
                best_beta = beta_guess
                best_theta = theta_guess
                best_lambdas = [l1, l2]

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_lambdas

    return best_beta, best_theta, best_cost
