import time
import numpy as np
import scipy as sp
import os
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/Users/jeanfeng/Documents/Research/descent_optimization')

import data_generation
from common import *
from convexopt_solvers import GroupedLassoProblemWrapper
from data_generation import sparse_groups

SIGNAL_NOISE_RATIO = 2

NUM_FEATURES = 250
NUM_NONZERO_FEATURES = 15
TRAIN_SIZE = 80
SEED = 10
TRUE_NUM_GROUPS = 3

def get_validation_cost(problem_wrapper, log_lambdas):
    start_time = time.time()
    lambdas = [10**float(l) for l in log_lambdas]
    betas = problem_wrapper.solve(lambdas)
    validation_cost = testerror_grouped(X_validate, y_validate, betas)
    runtime = time.time() - start_time
    return validation_cost, runtime


if __name__ == '__main__':
    np.random.seed(SEED)
    print "seed"
    sys.stdout.flush()

    TRAIN_SIZE = 60
    TOTAL_FEATURES = 300
    NUM_GROUPS = 30
    TRUE_GROUP_FEATURE_SIZES = [TOTAL_FEATURES / TRUE_NUM_GROUPS] * TRUE_NUM_GROUPS
    EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES = [TOTAL_FEATURES / NUM_GROUPS] * NUM_GROUPS

    beta_reals, X_train, y_train, X_validate, y_validate, X_test, y_test = sparse_groups(TRAIN_SIZE, TRUE_GROUP_FEATURE_SIZES)

    problem_wrapper = GroupedLassoProblemWrapper(X_train, y_train, EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES)


    for i in range(30):
        bashCommand = "python ../spearmint-master/spearmint-lite/spearmint-lite.py --method=GPEIOptChooser --method-args=noiseless=1 --grid-size=20000 bayesian_grouped_lasso"
        os.system(bashCommand)

        resfile = open('bayesian_grouped_lasso/results.dat','r')
        newlines = []
        for line in resfile.readlines():
            log_values = line.split()
            if len(log_values) < 30:
                continue
            val = log_values.pop(0)
            dur = log_values.pop(0)
            if (val == 'P'):
                print "running pending exp", log_values
                val, runtime = get_validation_cost(problem_wrapper, log_values)
                newlines.append(str(val) + " " + str(runtime) + " "
                                + " ".join(log_values)
                                + "\n")
            else:
                newlines.append(line)

        resfile.close()

        outfile = open('bayesian_grouped_lasso/results.dat','w')
        for line in newlines:
            outfile.write(line)
