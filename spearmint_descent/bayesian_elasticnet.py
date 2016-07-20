import numpy as np
import scipy as sp

import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/Users/jeanfeng/Documents/Research/descent_optimization')

import data_generation
from common import *
from convexopt_solvers import Lambda12ProblemWrapper

SIGNAL_NOISE_RATIO = 2

NUM_FEATURES = 250
NUM_NONZERO_FEATURES = 15
TRAIN_SIZE = 80
SEED = 10


def get_validation_cost(lambda1, lambda2):
    print "get_validation cost"
    sys.stdout.flush()
    np.random.seed(SEED)
    print "seed"
    sys.stdout.flush()
    beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test = data_generation.correlated(
        TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES, signal_noise_ratio=SIGNAL_NOISE_RATIO)
    print "data generated"
    sys.stdout.flush()

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)
    print "lambda1, lambda2", lambda1, lambda2
    sys.stdout.flush()
    beta_guess = problem_wrapper.solve(lambda1, lambda2)
    print "solved"
    sys.stdout.flush()
    current_cost = testerror(X_validate, y_validate, beta_guess)
    print "validation cost", current_cost
    sys.stdout.flush()
    return current_cost


def main(job_id, params):
    print 'Job #:', str(job_id)
    print params
    sys.stdout.flush()
    return get_validation_cost(params['lambda1'], params['lambda2'])
