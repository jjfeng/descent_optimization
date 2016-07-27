import getopt
import time
import sys
import traceback
import numpy as np
from multiprocessing import Pool

from sgl_hillclimb import SGL_Hillclimb
from sgl_neldermead import SGL_Nelder_Mead
from sgl_grid_search import SGL_Grid_Search
from sgl_spearmint import SGL_Spearmint
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult
from iteration_models import Simulation_Settings, Iteration_Data

from common import *

class SGL_Settings(Simulation_Settings):
    results_folder = "results/sgl"
    num_train = 10
    num_validate = 3
    num_test = 200
    num_features = 30
    expert_num_groups = 3
    true_num_groups = 3
    gs_lambdas1 = np.power(10, np.arange(-3, 1, 3.999/10))
    gs_lambdas2 = gs_lambdas1
    method = "HC"

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "expert_num_groups %d\n" % self.expert_num_groups
        obj_str += "num_features %d\n" % self.num_features
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
        obj_str += "nm_iters %d\n" % self.nm_iters
        print obj_str

    def get_true_group_sizes(self):
        assert(self.num_features % self.true_num_groups == 0)
        return [self.num_features / self.true_num_groups] * self.true_num_groups

    def get_expert_group_sizes(self):
        assert(self.num_features % self.expert_num_groups == 0)
        return [self.num_features / self.expert_num_groups] * self.expert_num_groups

#########
# MAIN FUNCTION
#########
def main(argv):
    seed = 10
    print "seed", seed
    np.random.seed(seed)
    num_threads = 1
    num_runs = 1

    try:
        opts, args = getopt.getopt(argv,"g:f:a:b:c:s:m:t:r:")
    except getopt.GetoptError:
        print "Bad argument given to sgl_eval.py"
        sys.exit(2)

    settings = SGL_Settings()
    for opt, arg in opts:
        if opt == '-g':
            settings.expert_num_groups = int(arg)
        elif opt == '-f':
            settings.num_features = int(arg)
        elif opt == '-a':
            settings.train_size = int(arg)
        elif opt == '-b':
            settings.validate_size = int(arg)
        elif opt == '-c':
            settings.test_size = int(arg)
        elif opt == "-s":
            settings.snr = float(arg)
        elif opt == "-m":
            assert(arg in METHODS)
            settings.method = arg
        elif opt == "-t":
            num_threads = int(arg)
        elif opt == "-r":
            num_runs = int(arg)

    print "TOTAL NUM RUNS %d" % num_runs
    settings.print_settings()
    sys.stdout.flush()

    data_gen = DataGenerator(settings)

    run_data = []
    for i in range(num_runs):
        observed_data = data_gen.sparse_groups()
        run_data.append(Iteration_Data(i, observed_data, settings))

    if settings.method != "SP" and num_threads > 1:
        print "Do multiprocessing"
        pool = Pool(num_threads)
        results = pool.map(fit_data_for_iter_safe, run_data)
    else:
        print "Avoiding multiprocessing"
        results = map(fit_data_for_iter_safe, run_data)

    method_results = MethodResults(settings.method)
    num_crashes = 0
    for r in results:
        if r is not None:
            method_results.append(r)
        else:
            num_crashes += 1
    print "==========TOTAL RUNS %d============" % method_results.get_num_runs()
    method_results.print_results()
    print "num crashes %d" % num_crashes

#########
# FUNCTIONS FOR CHILD THREADS
#########
def fit_data_for_iter_safe(iter_data):
    result = None
    try:
        result = fit_data_for_iter(iter_data)
    except Exception as e:
        print "Exception caught in iter %d: %s" % (iter_data.i, e)
        traceback.print_exc()
    return result

def fit_data_for_iter(iter_data):
    settings = iter_data.settings
    one_vec = np.ones(settings.expert_num_groups + 1)
    # Note that this produces quite different results from just having the latter set of lambda!
    # Hypothesis: warmstarts finds some good lambdas so that gradient descent will do quite well eventually.
    initial_lambdas_set = [one_vec, one_vec * 1e-1]
    method = iter_data.settings.method

    str_identifer = "%d_%d_%d_%d_%d_%d_%s_%d" % (
        settings.expert_num_groups,
        settings.num_features,
        settings.train_size,
        settings.validate_size,
        settings.test_size,
        settings.snr,
        method,
        iter_data.i
    )
    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    # set file buffer to zero so we can see progress
    with open(log_file_name, "w", buffering=0) as f:
        if method == "NM":
            algo = SGL_Nelder_Mead(iter_data.data, settings)
            algo.run(initial_lambdas_set, num_iters=settings.nm_iters, log_file=f)
        elif method == "GS":
            algo = SGL_Grid_Search(iter_data.data, settings)
            algo.run(lambdas1=settings.gs_lambdas1, lambdas2=settings.gs_lambdas2, log_file=f)
        elif method == "HC":
            algo = SGL_Hillclimb(iter_data.data, settings)
            algo.run(initial_lambdas_set, debug=False, log_file=f)
        elif method == "SP":
            algo = SGL_Spearmint(iter_data.data, str_identifer, settings)
            algo.run(settings.spearmint_numruns, log_file=f)
        sys.stdout.flush()
        method_res = create_method_result(iter_data.data, algo.fmodel)

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def create_method_result(data, algo, zero_threshold=1e-4):
    test_err = testerror_grouped(
        data.X_test,
        data.y_test,
        algo.best_model_params
    )

    beta_guess = np.concatenate(algo.best_model_params)

    guessed_nonzero_elems = np.where(get_nonzero_indices(beta_guess, threshold=zero_threshold))
    true_nonzero_elems = np.where(get_nonzero_indices(data.beta_real, threshold=zero_threshold))
    intersection = np.intersect1d(np.array(guessed_nonzero_elems), np.array(true_nonzero_elems))
    sensitivity = intersection.size / float(guessed_nonzero_elems[0].size) * 100

    beta_err = betaerror(data.beta_real, beta_guess)

    print "validation cost %f test_err %f sensitivity %f beta_err %f" % (
        algo.best_cost,
        test_err,
        sensitivity,
        beta_err,
    )
    return MethodResult(
        test_err=test_err,
        validation_err=algo.best_cost,
        beta_err=beta_err,
        runtime=algo.runtime,
        lambdas=algo.current_lambdas,
    )

if __name__ == "__main__":
    main(sys.argv[1:])
