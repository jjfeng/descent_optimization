import getopt
import time
import sys
import traceback
import numpy as np
from multiprocessing import Pool

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb
from sparse_add_models_neldermead import Sparse_Add_Model_Nelder_Mead
from sparse_add_models_grid_search import Sparse_Add_Model_Grid_Search
from sparse_add_models_spearmint import Sparse_Add_Model_Spearmint
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult
from iteration_models import Simulation_Settings, Iteration_Data

from common import *

METHODS = ["NM", "HC", "GS", "SP"]

class Sparse_Add_Models_Settings(Simulation_Settings):
    results_folder = "results/sparse_add_models"
    num_funcs = 2
    num_zero_funcs = 2
    gs_lambdas1 = np.power(10, np.arange(-5, 2, 6.999/10))
    gs_lambdas2 = gs_lambdas1
    method = "HC"

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "num_funcs %d\n" % self.num_funcs
        obj_str += "num_zero_funcs %d\n" % self.num_zero_funcs
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
        obj_str += "nm_iters %d\n" % self.nm_iters
        print obj_str

########
# FUNCTIONS FOR ADDITIVE MODEL
########
def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(9 * np.sin(x*3)) # 2

def big_cos_sin(x):
    return identity_fcn(6 * (np.cos(x * 1.25) + np.sin(x/2 + 0.5)))

def crazy_down_sin(x):
    return identity_fcn(x * np.sin(x) - x)

def pwr_small(x):
    return identity_fcn(np.power(x/2,2) - 10)

def const_zero(x):
    return np.zeros(x.shape)

#########
# MAIN FUNCTION
#########
def main(argv):
    seed = 10
    print "seed", seed
    np.random.seed(seed)
    num_threads = 1
    num_runs = 30

    try:
        opts, args = getopt.getopt(argv,"f:z:a:b:c:s:m:t:r:")
    except getopt.GetoptError:
        sys.exit(2)

    settings = Sparse_Add_Models_Settings()
    for opt, arg in opts:
        if opt == '-f':
            settings.num_funcs = int(arg)
        elif opt == '-z':
            settings.num_zero_funcs = int(arg)
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

    SMOOTH_FCNS = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    assert(settings.num_funcs <= len(SMOOTH_FCNS))
    smooth_fcn_list = SMOOTH_FCNS[:settings.num_funcs] + [const_zero] * settings.num_zero_funcs
    data_gen = DataGenerator(settings)

    run_data = []
    for i in range(num_runs):
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)
        run_data.append(Iteration_Data(i, observed_data, settings))

    if settings.method != "SP":
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
    initial_lambdas = np.ones(1 + settings.num_funcs + settings.num_zero_funcs)
    initial_lambdas[0] = 5
    method = iter_data.settings.method

    str_identifer = "%d_%d_%d_%d_%d_%d_%s_%d" % (
        settings.num_funcs,
        settings.num_zero_funcs,
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
            algo = Sparse_Add_Model_Nelder_Mead(iter_data.data)
            algo.run(initial_lambdas, num_iters=settings.nm_iters, log_file=f)
        elif method == "GS":
            algo = Sparse_Add_Model_Grid_Search(iter_data.data)
            algo.run(lambdas1=settings.gs_lambdas1, lambdas2=settings.gs_lambdas2, log_file=f)
        elif method == "HC":
            algo = Sparse_Add_Model_Hillclimb(iter_data.data)
            algo.run([initial_lambdas], debug=False, log_file=f)
        elif method == "SP":
            algo = Sparse_Add_Model_Spearmint(iter_data.data, str_identifer)
            algo.run(settings.spearmint_numruns, log_file=f)
        sys.stdout.flush()
        method_res = create_method_result(iter_data.data, algo.fmodel)

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def create_method_result(data, algo):
    test_err = testerror_sparse_add_smooth(
        data.y_test,
        data.test_idx,
        algo.best_model_params
    )
    print "validation cost", algo.best_cost, "test_err", test_err
    return MethodResult(
        test_err=test_err,
        validation_err=algo.best_cost,
        runtime=algo.runtime,
        lambdas=algo.current_lambdas
    )

if __name__ == "__main__":
    main(sys.argv[1:])
