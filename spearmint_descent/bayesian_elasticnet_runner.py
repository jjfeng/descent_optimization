import numpy as np
import scipy as sp
import os
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
    np.random.seed(SEED)
    print "seed"
    sys.stdout.flush()
    beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test = data_generation.correlated(
        TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES, signal_noise_ratio=SIGNAL_NOISE_RATIO)

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)
    print "lambda1, lambda2", lambda1, lambda2
    sys.stdout.flush()
    beta_guess = problem_wrapper.solve(lambda1, lambda2)
    current_cost = testerror(X_validate, y_validate, beta_guess)
    return current_cost


if __name__ == '__main__':
    for i in range(30):
        bashCommand = "python ../spearmint-master/spearmint-lite/spearmint-lite.py --method=GPEIOptChooser --method-args=use_multiprocessing=0 bayesian_elasticnet"
        os.system(bashCommand)

        resfile = open('bayesian_elasticnet/results.dat','r')
        newlines = []
        for line in resfile.readlines():
            values = line.split()
            if len(values) < 3:
                continue
            val = values.pop(0)
            dur = values.pop(0)
            print "lambda12", float(values[0]), float(values[1])
            if (val == 'P'):
                val = get_validation_cost(float(values[0]), float(values[1]))
                newlines.append(str(val) + " 0 "
                                + str(float(values[0])) + " "
                                + str(float(values[1])) + "\n")
            else:
                newlines.append(line)

        resfile.close()
        outfile = open('bayesian_elasticnet/results.dat','w')
        for line in newlines:
            outfile.write(line)
