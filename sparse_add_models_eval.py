import getopt
import time
import sys
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

NUM_RUNS = 8
METHODS = ["NM", "HC", "GS", "SP"]

class Sparse_Add_Models_Settings(Simulation_Settings):
    num_funcs = 2
    num_zero_funcs = 2
    gs_lambdas1 = np.power(10, np.arange(-5, 2, 6.999/10))
    gs_lambdas2 = gs_lambdas1
    spearmint_numruns = 100
    nm_iters = 80
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

def main(argv):
    seed = 10
    print "seed", seed
    np.random.seed(seed)
    num_threads = 1

    try:
        opts, args = getopt.getopt(argv,"f:z:a:b:c:s:m:t:")
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
            settings.num_threads = int(arg)

    print settings.print_settings()
    sys.stdout.flush()

    SMOOTH_FCNS = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    assert(settings.num_funcs <= len(SMOOTH_FCNS))
    smooth_fcn_list = SMOOTH_FCNS[:settings.num_funcs] + [const_zero] * settings.num_zero_funcs
    data_gen = DataGenerator(settings)

    pool = Pool(num_threads)
    run_data = []
    for i in range(NUM_RUNS):
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)
        run_data.append(Iteration_Data(observed_data, settings))

    results = pool.map(fit_data_for_iter, run_data)
    method_results = MethodResults(settings.method)
    for r in results:
        method_results.append(r)
    print "==========TOTAL RUNS %d============" % NUM_RUNS
    method_results.print_results()

def fit_data_for_iter(iter_data):
    settings = iter_data.settings
    initial_lambdas = np.ones(1 + settings.num_funcs + settings.num_zero_funcs)
    initial_lambdas[0] = 30
    method = iter_data.settings.method

    if method == "NM":
        algo = Sparse_Add_Model_Nelder_Mead(iter_data.data)
        algo.run(initial_lambdas, num_iters=settings.nm_iters)
    elif method == "GS":
        algo = Sparse_Add_Model_Grid_Search(iter_data.data)
        algo.run(gs_lambdas1, gs_lambdas2)
    elif method == "HC":
        algo = Sparse_Add_Model_Hillclimb(iter_data.data)
        algo.run([initial_lambdas], debug=False)
    elif method == "SP":
        sp_identifer = "%d_%d_%d_%d_%d_%d" % (
            settings.num_funcs,
            settings.num_zero_funcs,
            settings.train_size,
            settings.validate_size,
            settings.test_size,
            settings.snr
        )
        algo = Sparse_Add_Model_Spearmint(iter_data.data, sp_identifer)
        algo.run(spearmint_numruns)
    sys.stdout.flush()
    return create_method_result(iter_data.data, algo.fmodel)

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
