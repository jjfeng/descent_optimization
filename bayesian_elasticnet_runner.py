import sys
import time

import numpy as np
import scipy as sp

import data_generation
from common import *
from convexopt_solvers import Lambda12ProblemWrapper

NUM_RUNS = 15
RESULT_FOLDER = "spearmint_descent/bayesian_elasticnet"

def run(X_train, y_train, X_validate, y_validate):
    best_cost = 100000
    total_time = 0
    start_time = time.time()

    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)

    total_time += time.time() - start_time

    result_file_name = "results_%d.dat" % start_time
    result_file = "%s/%s" % (RESULT_FOLDER, result_file_name)
    for i in range(NUM_RUNS):
        start_time = time.time()
        # Run spearmint to get next experiment parameters
        run_spearmint_command(result_file_name, RESULT_FOLDER)

        # Find new experiment
        with open(result_file,'r') as resfile:
            newlines = []
            for line in resfile.readlines():
                values = line.split()
                if len(values) < 3:
                    continue
                val = values.pop(0)
                dur = values.pop(0)
                lambda1 = 10**float(values[0])
                lambda2 = 10**float(values[1])
                print "lambda12", lambda1, lambda2
                if (val == 'P'):
                    # P means pending experiment to run
                    # Run experiment
                    beta_guess = problem_wrapper.solve(lambda1, lambda2)
                    current_cost = testerror(X_validate, y_validate, beta_guess)

                    if best_cost > current_cost:
                        best_cost = current_cost
                        best_betas = beta_guess

                    newlines.append(str(current_cost) + " 0 "
                                    + " ".join(values) + "\n")
                else:
                    # Otherwise these are previous experiment results
                    newlines.append(line)
            total_time += time.time() - start_time

        # Don't record time spent on writing files?
        with open(result_file,'w') as resfile:
            resfile.writelines(newlines)

    return best_betas, total_time
