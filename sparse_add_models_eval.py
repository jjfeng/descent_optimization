import getopt
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult

from common import *

np.random.seed(1)

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
    num_funcs = 3
    num_zero_funcs = 3
    train_size = 30
    validate_size = 30
    test_size = 30
    snr = 2

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
    smooth_fcn_list = SMOOTH_FCNS[:num_funcs] + [const_zero] * num_zero_funcs

    data_gen = DataGenerator(train_size, validate_size, test_size, feat_range=[0,1], snr=snr)
    observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)

    hc_results = MethodResults("Hillclimb")

    hc_algo = Sparse_Add_Model_Hillclimb(observed_data)
    initial_lambdas = np.ones(1 + num_funcs + num_zero_funcs)
    hc_algo.run(initial_lambdas)

    print "===========RUN ============"
    hc_results.print_results()

if __name__ == "__main__":
    main(sys.argv[1:])
