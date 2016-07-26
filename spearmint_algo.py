import sys
import time

import numpy as np
import scipy as sp

import data_generation
from common import *
from fitted_model import Fitted_Model

class Spearmint_Algo:
    MAX_COST = 1e6

    def __init__(self, data, folder_suffix):
        self.data = data
        # Call problem wrapper first to determine number of lambdas
        self._create_problem_wrapper()
        self._check_make_configs(folder_suffix)
        self.result_file = "%s/results.dat" % self.result_folder

    def run(self, num_runs, log_file=None):
        self.log_file = log_file
        # Run spearmint to get next experiment parameters
        self.fmodel = Fitted_Model(self.num_lambdas)

        # Find new experiment
        best_cost = None
        runtime = 0
        for i in range(num_runs):
            start_time = time.time()
            self.log("%s: iter %d" % (self.method_label, i))
            self.run_spearmint_command(self.result_folder)

            with open(self.result_file,'r') as resfile:
                newlines = []
                for line in resfile.readlines():
                    values = line.split()
                    if len(values) < 3:
                        continue
                    val = values.pop(0)
                    dur = values.pop(0)
                    lambdas = np.array([10**float(v) for v in values[:self.num_lambdas]])
                    if (val == 'P'):
                        # P means pending experiment to run
                        # Run experiment
                        self.log("lambdas %s" % lambdas)
                        model_params = self._solve_problem(lambdas)
                        if model_params is None:
                            current_cost = self.MAX_COST
                        else:
                            current_cost = self.get_validation_cost(model_params)

                        if best_cost is None or best_cost > current_cost:
                            best_cost = current_cost
                            self.fmodel.update(lambdas, model_params, current_cost)
                            self.log("fmodel: %s" % self.fmodel)

                        newlines.append(str(current_cost) + " 0 "
                                        + " ".join(values) + "\n")
                    else:
                        # Otherwise these are previous experiment results
                        newlines.append(line)

            runtime += time.time() - start_time
            # Don't record time spent on writing files?
            with open(self.result_file, 'w') as resfile:
                resfile.writelines(newlines)

            sys.stdout.flush()

        self.fmodel.set_runtime(runtime)
        self.log("%s: runtime %s" % (self.method_label, runtime))
        self.log("fmodel: %s" % self.fmodel)

        # VERY IMPORTANT to clean spearmint results
        self.run_spearmint_clean(self.result_folder)

    def log(self, log_str):
        if self.log_file is None:
            print log_str
        else:
            self.log_file.write(log_str)

    @staticmethod
    def run_spearmint_command(experiment_folder, use_multiprocessing=True, gridsize=20000):
        multiprocessing_option = ""
        if not use_multiprocessing:
            multiprocessing_option = "--method-args=use_multiprocessing=0"
        cmd = "python2.7 spearmint-master/spearmint-lite/spearmint-lite.py --method=GPEIOptChooser --method-args=noiseless=1 --grid-size=%d" % gridsize
        os.system("%s %s %s" % (cmd, multiprocessing_option, experiment_folder))

    @staticmethod
    def run_spearmint_clean(experiment_folder):
        cmd = "rm %s/chooser* %s/results.dat" % (experiment_folder, experiment_folder)
        os.system(cmd)

    @staticmethod
    def _create_config_string(num_lambdas):
        return '{"lambda": {"name": "lambda","type": "float","size":%d,"min":  -6,"max":  2}}' % num_lambdas
