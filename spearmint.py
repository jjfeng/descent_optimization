import sys
import time

import numpy as np
import scipy as sp

import data_generation
from common import *
from fitted_model import Fitted_Model

class Spearmint_Algo:
    MAX_COST = 1e6

    def __init__(self, data):
        self.data = data
        self.result_file = "%s/results.dat" % self.result_folder
        self._create_problem_wrapper()

    def run(self, num_runs):
        start_time = time.time()
        # Run spearmint to get next experiment parameters
        run_spearmint_command(self.result_folder)

        self.fmodel = Fitted_Model(self.num_lambdas)

        # Find new experiment
        for i in range(num_runs):
            with open(self.result_file,'r') as resfile:
                newlines = []
                for line in resfile.readlines():
                    values = line.split()
                    if len(values) < 3:
                        continue
                    val = values.pop(0)
                    dur = values.pop(0)
                    lambdas = np.array(values[:self.num_lambdas])
                    if (val == 'P'):
                        # P means pending experiment to run
                        # Run experiment
                        model_params = self.problem_wrapper.solve(lambdas)
                        current_cost = self.get_validation_cost(model_params)

                        if best_cost > current_cost:
                            best_cost = current_cost
                            self.fmodel.update(lambdas, model_params, current_cost)

                        newlines.append(str(current_cost) + " 0 "
                                        + " ".join(values) + "\n")
                    else:
                        # Otherwise these are previous experiment results
                        newlines.append(line)
                runtime += time.time() - start_time

                # Don't record time spent on writing files?
                with open(self.result_file, 'w') as resfile:
                    resfile.writelines(newlines)

            self.fmodel.set_runtime(runtime)

            # VERY IMPORTANT to clean spearmint results
            run_spearmint_clean(self.result_folder)
