import getopt
import time
import sys
import numpy as np

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb
from sparse_add_models_neldermead import Sparse_Add_Model_Nelder_Mead
from sparse_add_models_grid_search import Sparse_Add_Model_Grid_Search
from sparse_add_models_spearmint import Sparse_Add_Model_Spearmint
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult

from common import *

np.random.seed(1)
NUM_RUNS = 30

def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(9 * np.sin(x*2))

def big_cos_sin(x):
    return identity_fcn(6 * (np.cos(x * 1.25) + np.sin(x/2 + 0.5)))

def crazy_down_sin(x):
    return identity_fcn(x * np.sin(x) - x)

def pwr_small(x):
    return identity_fcn(np.power(x/2,2) - 10)

def const_zero(x):
    return np.zeros(x.shape)


def main(argv):
    num_funcs = 2
    num_zero_funcs = 2
    train_size = 100
    validate_size = 50
    test_size = 50
    snr = 2
    gs_lambdas1 = [0.125, 0.25, 0.5, 1, 2]
    gs_lambdas2 = gs_lambdas1
    spearmint_numruns = 15
    seed = 10

    np.random.seed(seed)

    try:
        opts, args = getopt.getopt(argv,"f:z:a:b:c:s:")
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-f':
            num_funcs = int(arg)
        elif opt == '-z':
            num_zero_funcs = int(arg)
        elif opt == '-a':
            train_size = int(arg)
        elif opt == '-b':
            validate_size = int(arg)
        elif opt == '-c':
            test_size = int(arg)
        elif opt == "-s":
            snr = float(snr)

    print "num_funcs", num_funcs
    print "num_zero_funcs", num_zero_funcs
    print "t/v/t size", train_size, validate_size, test_size
    print "snr", snr
    print "seed", seed
    sys.stdout.flush()

    SMOOTH_FCNS = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    assert(num_funcs <= len(SMOOTH_FCNS))
    smooth_fcn_list = SMOOTH_FCNS[:num_funcs] + [const_zero] * num_zero_funcs
    data_gen = DataGenerator(train_size, validate_size, test_size, feat_range=[-5,5], snr=snr)

    hc_results = MethodResults("Hillclimb")
    nm_results = MethodResults("NelderMead")
    gs_results = MethodResults("Gridsearch")
    sp_results = MethodResults("Spearmint")
    for i in range(NUM_RUNS):
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)

        initial_lambdas = np.ones(1 + num_funcs + num_zero_funcs)

        # nm_algo = Sparse_Add_Model_Nelder_Mead(observed_data)
        # nm_algo.run(initial_lambdas[:2])
        # nm_results.append(create_method_result(observed_data, nm_algo.fmodel))
        # sys.stdout.flush()
        #
        gs_algo = Sparse_Add_Model_Grid_Search(observed_data)
        gs_algo.run(gs_lambdas1, gs_lambdas2)
        gs_results.append(create_method_result(observed_data, gs_algo.fmodel))
        sys.stdout.flush()

        hc_algo = Sparse_Add_Model_Hillclimb(observed_data)
        hc_algo.run([initial_lambdas, initial_lambdas * 0.1], debug=False) #False)
        hc_results.append(create_method_result(observed_data, hc_algo.fmodel))
        sys.stdout.flush()

        # sp_algo = Sparse_Add_Model_Spearmint(observed_data)
        # sp_algo.run(spearmint_numruns)
        # sp_results.append(create_method_result(observed_data, sp_algo.fmodel))

        print "===========RUN %d ============" % i
        hc_results.print_results()
        nm_results.print_results()
        gs_results.print_results()
        sp_results.print_results()
        sys.stdout.flush()

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
        runtime=algo.runtime
    )

if __name__ == "__main__":
    main(sys.argv[1:])
