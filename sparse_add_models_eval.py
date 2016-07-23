import getopt
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb
from sparse_add_models_neldermead import Sparse_Add_Model_Nelder_Mead
from sparse_add_models_grid_search import Sparse_Add_Model_Grid_Search
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult

from common import *

np.random.seed(1)
NUM_RUNS = 1

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
    num_funcs = 5
    num_zero_funcs = 5
    train_size = 100
    validate_size = 50
    test_size = 50
    snr = 2
    gs_lambdas1 = [0.1, 1]
    gs_lambdas2 = [1]

    np.random.seed(10)

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

    SMOOTH_FCNS = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    assert(num_funcs <= len(SMOOTH_FCNS))
    smooth_fcn_list = SMOOTH_FCNS[:num_funcs] + [const_zero] * num_zero_funcs
    data_gen = DataGenerator(train_size, validate_size, test_size, feat_range=[0,1], snr=snr)

    hc_results = MethodResults("Hillclimb")
    nm_results = MethodResults("NelderMead")
    gs_results = MethodResults("Gridsearch")
    for i in range(NUM_RUNS):
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)

        initial_lambdas = np.ones(1 + num_funcs + num_zero_funcs)

        # nm_algo = Sparse_Add_Model_Nelder_Mead(observed_data)
        # nm_algo.run(initial_lambdas[:2])
        # nm_results.append(create_method_result(observed_data, nm_algo.fmodel))

        gs_algo = Sparse_Add_Model_Grid_Search(observed_data)
        gs_algo.run(gs_lambdas1, gs_lambdas2)
        gs_results.append(create_method_result(observed_data, gs_algo.fmodel))

        hc_algo = Sparse_Add_Model_Hillclimb(observed_data)
        hc_algo.run(initial_lambdas, debug=False) #True)
        hc_results.append(create_method_result(observed_data, hc_algo.fmodel))

        print "===========RUN ============"
        hc_results.print_results()
        nm_results.print_results()

def create_method_result(data, algo):
    test_err = testerror_sparse_add_smooth(
        data.y_test,
        data.test_idx,
        algo.current_model_params
    )
    print "validation cost", algo.current_cost, "test_err", test_err
    return MethodResult(
        test_err=test_err,
        validation_err=algo.current_cost,
        runtime=algo.runtime
    )

if __name__ == "__main__":
    main(sys.argv[1:])
